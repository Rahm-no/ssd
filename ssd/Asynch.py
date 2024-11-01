import subprocess
import threading
import time
import torch
import numpy as np
from queue import Full, Empty
from torch.utils.data import DataLoader
import psutil
import csv
import utils
import multiprocessing as mp

# Global stop event for monitoring
data_stop_event = mp.Event()
class DataProducer(mp.Process):
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
    
    def run(self):
        try:
            while not data_stop_event.is_set():  # Stop if the event is set
                batch = []
                for idx in self.indices:
                    if data_stop_event.is_set():
                        break

                    # Check if idx is a list
                    if isinstance(idx, list):
                        for sub_idx in idx:
                            if data_stop_event.is_set():
                                break

                            print(f"Producer {self.pid} processing sub-index {sub_idx}")
                            sample = self.dataset[sub_idx]
                            batch.append(sample)

                            if len(batch) == self.batch_size:
                                self.put_batch(batch)
                                batch = []

                            if data_stop_event.is_set():
                                break
                    else:
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
                    print('add to Queue', self.queue.qsize())
                    self.queue.put((images, labels), timeout=1)

                    break
                except Full:
                    if data_stop_event.is_set():
                        return  # Exit if stop event is set
                    print("Queue is full, retrying...")
                    time.sleep(0.5)  # Wait before retrying
        except Exception as e:
            print(f"Error in producer {self.pid}: {e}")

    def _stop(self):
        data_stop_event.set()


class AsynchronousLoader(DataLoader):
    def __init__(self, queue, dataset, device, shards, rank, batch_size, shuffle, pin_memory, num_workers, queue_size=20, sampler=None, drop_last=True):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=drop_last,
            sampler=sampler  # Pass sampler here
        )
        self.queue_size = queue_size
        self.queue = queue
        self.device = device
        self.shards = shards
        self.num_workers = num_workers
        self.rank = rank
        self.queue_sizes = []
        self.epoch_batches = len(self.dataset) // (self.batch_size * self.shards)  # Total batches per epoch
        print("batch size used here", self.batch_size)

        self.shuffle = shuffle

        self.indices = list(self.sampler) # it gives the indices per gpu
        self.start_threads()


    def start_threads(self):
        self.producers = []
        total_samples = len(self.indices)
        indices_per_producer = max(1, total_samples // self.num_workers)  # Ensure at least one index per producer
        start_idx = 0

        for i in range(self.num_workers):
            end_idx = min(start_idx + indices_per_producer, total_samples)
            producer_indices = self.indices[start_idx:end_idx]
            start_idx = end_idx

            if not producer_indices:  # Break if no indices are available
                break

            print(f"[GPU Rank {self.rank}] Producer {i} assigned indices: {producer_indices}")
            
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
                print('get from Queue before', self.queue.qsize())

                batch = self.queue.get(timeout=1)
                images, targets = batch
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
             
                self.batches_processed += 1
                print('get from Queue after', self.queue.qsize())


                return images, targets

            except Empty:
            # If the queue is empty and the stop event is set, break the loop
                if data_stop_event.is_set():
                    print("Stopping iteration due to stop event.")
                    raise StopIteration  # End iteration if threads have stopped
                else:
                    time.sleep(0.1)  # Continue retrying if the stop event is no









##############


    # def run(self):
    #     try:
    #         while not data_stop_event.is_set():  # Stop if the event is set
    #             batch = []
    #             for idx in self.indices:
    #                 # Check if idx is a list
    #                 if isinstance(idx, list):
    #                     # If idx is a list, iterate through its elements
    #                     for sub_indx in idx:
    #                         if data_stop_event.is_set():
    #                             break  # Exit if stop event is triggered

    #                         print(f"Producer {self.pid} processing sub-index {sub_indx}")
    #                         sample = self.dataset[sub_indx]  # Access the dataset with sub_indx
    #                         batch.append(sample)

    #                         if len(batch) == self.batch_size:
    #                             self.put_batch(batch)  # Process the full batch
    #                             batch = []  # Reset the batch

    #                         if data_stop_event.is_set():
    #                             break  # Exit if stop event is triggered

    #                 else:  # idx is assumed to be a single index
    #                     if data_stop_event.is_set():
    #                         break  # Exit if stop event is triggered

    #                     print(f"Producer {self.pid} processing index {idx}")
    #                     sample = self.dataset[idx]  # Access the dataset with idx
    #                     batch.append(sample)

    #                     if len(batch) == self.batch_size:
    #                         self.put_batch(batch)  # Process the full batch
    #                         batch = []  # Reset the batch

    #             # After processing all indices, check for any remaining samples
    #             if batch and not data_stop_event.is_set():
    #                 print(f"Producer {self.pid} has remaining batch")
    #                 self.put_batch(batch)  # Process remaining samples

    #     except Exception as e:
    #         print(f"Error in producer {self.pid}: {e}")  # Log any exceptions
    #     finally:
    #         print(f"Producer {self.pid} exiting.")  # Exit message
