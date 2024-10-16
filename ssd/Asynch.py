import subprocess
import threading
import time
import torch
import numpy as np
from queue import Full, Empty
from torch.utils.data import DataLoader
import psutil
import csv

# Global stop event for monitoring
data_stop_event = threading.Event()
class DataProducer(threading.Thread):
    def __init__(self, queue, dataset, batch_size, shuffle, pin_memory, device, queue_size, indices, sampler, num_workers):
        super().__init__()
        self.queue = queue
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.device = device
        self.indices = indices
        self.sampler = sampler
        self.num_workers = num_workers
        self.pid = threading.get_ident()  # Get thread ID

    def run(self):
        try:
            while not data_stop_event.is_set():  # Stop if the event is set
                batch = []
                for idx in self.indices:
                    if data_stop_event.is_set():
                        break

                    print(f"Producer {self.pid} processing index {idx}")
                    sample = self.dataset[idx]
                    batch.append(sample)

                    if len(batch) == self.batch_size:
                        self.put_batch(batch)
                        batch = []

                    if data_stop_event.is_set():
                        break

                if batch and not data_stop_event.is_set():
                    print(f"Producer {self.pid} has remaining batch")
                    self.put_batch(batch)
                
        except Exception as e:
            print(f"Error in producer {self.pid}: {e}")
        finally:
            print(f"Producer {self.pid} exiting.")



    def put_batch(self, batch):
        try:
            images, labels = zip(*batch)
            
            while not data_stop_event.is_set():
                try:
                    # Try to put data in queue
                    self.queue.put((images, labels), timeout=1)
                    break
                except Full:
                    if data_stop_event.is_set():
                        return  # Exit if stop event is set
                    print("Queue is full, retrying...")
                    time.sleep(0.5)  # Wait before retrying
        except Exception as e:
            print(f"Error in producer {self.pid}: {e}")

    # def stop(self):
    #     self.stop_flag = True  # Set the stop flag



class AsynchronousLoader(DataLoader):
    def __init__(self,queue, dataset, device, shards, rank, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, queue_size=20, drop_last=True, sampler=None):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=0, drop_last=drop_last, sampler=sampler)
        self.queue_size = queue_size
        self.queue = queue
        self.device = device
        self.shards = shards
        self.num_workers = num_workers
        self.rank = rank
        self.queue_sizes = []
        self.epoch_batches = len(self.dataset) // (self.batch_size * self.shards)  # Total batches per epoch
        
        self.shuffle = shuffle

        self.indices = list(self.sampler) # it gives the indices per gpu
        self.start_threads()

    def start_threads(self):
        self.producers = []
        total_samples = len(self.indices)
        indices_per_producer = total_samples // self.num_workers
        start_idx = 0

        for i in range(self.num_workers):
            end_idx = start_idx + indices_per_producer
            producer_indices = self.indices[start_idx:end_idx]
            start_idx = end_idx
            producer = DataProducer(self.queue, self.dataset, self.batch_size, self.shuffle,
                                    self.pin_memory, self.device, self.queue_size,
                                    producer_indices, self.sampler, self.num_workers)
            producer.start()
            self.producers.append(producer)

    def stop_threads(self):
        print('STOPPING THREADS ')
        data_stop_event.set()  # Signal threads to stop
        for producer in self.producers:
            if producer.is_alive():
                print(f"Joining producer {producer.pid}")
                producer.join(timeout=5)  # Add a timeout to avoid hanging forever
                if producer.is_alive():
                    print(f"Producer {producer.pid} did not exit, forcefully terminating...")
                    producer._stop()  # Force stop if the producer doesn't exit cleanly (not recommended, but an option)


        # Clean up the multiprocessing queue if using multiprocessing.Queue
        self.queue.close()
        self.queue.join_thread()
        print("Queue closed and joined.")

    def __iter__(self):
        self.batches_processed = 0  # Reset batch counter for the new epoch
        return self

    def __next__(self):
        print("batch_processed", self.batches_processed)
        if self.batches_processed >= self.epoch_batches:
            print('RAISE stopITERATION')
            raise StopIteration  # End of epoch

        while True:
            try:
                batch = self.queue.get(timeout=1)
                self.batches_processed += 1

                return batch

            except Empty:
            # If the queue is empty and the stop event is set, break the loop
                if data_stop_event.is_set():
                    print("Stopping iteration due to stop event.")
                    raise StopIteration  # End iteration if threads have stopped
                else:
                    time.sleep(0.1)  # Continue retrying if the stop event is no