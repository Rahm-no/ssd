import subprocess
import threading
import time
import torch
import os
import numpy as np
from queue import Full, Empty
import multiprocessing as mp
from torch.utils.data import DataLoader
import psutil
import csv

# Global stop event for monitoring
data_stop_event = mp.Event()
class DataProducer(mp.Process):
    def __init__(self, queue, dataset, batch_size, shuffle, pin_memory, device, queue_size, indices, sampler, num_workers,rank):
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
        self.rank = rank




    def run(self):
        print(f"[GPU Rank {self.rank} | Producer {self.pid}] Starting.")
        try:
            while not data_stop_event.is_set():
                batch = []
                for idx in self.indices:
                    if data_stop_event.is_set():
                        break

                    print(f"[GPU Rank {self.rank} | Producer {self.pid}] Processing index {idx}")
                    sample = self.dataset[idx]
                    batch.append(sample)

                    if len(batch) == self.batch_size:
                        self.put_batch(batch)
                        batch = []

                    if data_stop_event.is_set():
                        break

                if batch and not data_stop_event.is_set():
                    print(f"[GPU Rank {self.rank} | Producer {self.pid}] Has remaining batch")
                    self.put_batch(batch)

            print(f"[GPU Rank {self.rank} | Producer {self.pid}] Exiting.")
            
        except Exception as e:
            print(f"[GPU Rank {self.rank} | Producer {self.pid}] Error: {e}")




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

    def _stop(self):
        data_stop_event.set()
       



class AsynchronousLoader(DataLoader):
    def __init__(self, queue, dataset, device, shards, rank, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, queue_size=20, drop_last=True, sampler=None):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=0, drop_last=drop_last, sampler=sampler)
        self.queue_size = queue_size
        self.queue = queue
      
        self.device = device
        self.shards = shards
        self.num_workers = num_workers
        self.rank = rank

        self.epoch_batches = len(self.dataset) // (self.batch_size * self.shards)  # Total batches per epoch
        
        self.shuffle = shuffle

        self.indices = list(self.sampler) # it gives the indices per gpu
        print(f"GPU Rank {self.rank} is assigned {len(self.indices)} samples.")
        
        print(f"Created queue for GPU Rank {self.rank}.")

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
                                    producer_indices, self.sampler, self.num_workers,self.rank)

            producer.start()
            self.producers.append(producer)







        # total_samples = len(self.indices)
        # indices_per_producer = total_samples // self.num_workers
        # remainder = total_samples % self.num_workers  # Handle extra samples

        # start_idx = 0
        # for i in range(self.num_workers):
        #     # Give one extra sample to the first 'remainder' producers
        #     end_idx = start_idx + indices_per_producer + (1 if i < remainder else 0)
        #     producer_indices = self.indices[start_idx:end_idx]
        #     start_idx = end_idx
        #     producer = DataProducer(self.queue, self.dataset, self.batch_size, self.shuffle,
        #                             self.pin_memory, self.device, self.queue_size,
        #                             producer_indices, self.sampler, self.num_workers)
            
        #     # Log the indices assigned to each producer
        #     print(f"[GPU Rank {self.rank}] Producer {i} assigned indices {producer_indices}, Total: {len(producer_indices)} samples.")
            
        #     producer.start()
        #     self.producers.append(producer)
    def stop_threads(self):
        print('STOPPING THREADS ')
        data_stop_event.set()  # Signal threads to stop
        for producer in self.producers:
            if producer.is_alive():
                print(f"Joining producer")
                producer.join(timeout=5)  # Add a timeout to avoid hanging forever
                if producer.is_alive():
                    print(f"Producer did not exit, forcefully terminating...")
                    producer._stop()  # Force stop if the producer doesn't exit cleanly (not recommended, but an option)


        # Clean up the multiprocessing queue if using multiprocessing.Queue
        self.queue.close()
        self.queue.join_thread()
        print("Queue closed and joined.")

    def __iter__(self):
        self.batches_processed = 0  # Reset batch counter for the new epoch
        return self
    def __next__(self):
        print("Batch processed:", self.batches_processed, "for rank:", self.rank)
        
        if self.batches_processed >= self.epoch_batches:
            print('RAISE StopIteration')
            raise StopIteration  # End of epoch

        while True:
            try:
                # Print the queue size specific to this GPU rank
                print(f"Queue size for GPU {self.rank}: {self.queue.qsize()} samples")

                # Get a batch from the queue
                batch = self.queue.get(timeout=1)
                self.batches_processed += 1

                # Print the processed batch info
                print(f"[GPU Rank {self.rank}] Processed batch {self.batches_processed}.")
                
                return batch

            except Empty:
                # If the queue is empty and the stop event is set, break the loop
                if data_stop_event.is_set():
                    print("Stopping iteration due to stop event.")
                    break

