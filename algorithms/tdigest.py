import math
import bisect
from typing import List, Tuple


class Centroid:
    def __init__(self, mean: float, count: int):
        self.mean = mean
        self.count = count
    
    def add(self, value: float, count: int = 1):
        total_count = self.count + count
        self.mean = (self.mean * self.count + value * count) / total_count
        self.count = total_count


class TDigest:
    
    def __init__(self, delta: float = 0.01, k: int = 25):
        self.delta = delta
        self.k = k
        self.centroids: List[Centroid] = []
        self.count = 0
        self._cumulative_counts = []
    
    def update(self, value: float, count: int = 1) -> None:
        if not self.centroids:
            self.centroids.append(Centroid(value, count))
            self.count = count
            self._update_cumulative()
            return
        
        # Find insertion point
        insertion_index = bisect.bisect_left([c.mean for c in self.centroids], value)
        
        # Try to merge with nearby centroids
        merged = False
        
        # Check left neighbor
        if insertion_index > 0:
            left_centroid = self.centroids[insertion_index - 1]
            if self._can_merge(left_centroid, value):
                left_centroid.add(value, count)
                merged = True
        
        # Check right neighbor if not merged
        if not merged and insertion_index < len(self.centroids):
            right_centroid = self.centroids[insertion_index]
            if self._can_merge(right_centroid, value):
                right_centroid.add(value, count)
                merged = True
        
        # Create new centroid if no merge possible
        if not merged:
            self.centroids.insert(insertion_index, Centroid(value, count))
        
        self.count += count
        
        # Compress if we have too many centroids
        if len(self.centroids) > self.k:
            self._compress()
        
        self._update_cumulative()
    
    def quantile(self, q: float) -> float:
        if not self.centroids:
            return 0.0
        
        if q <= 0:
            return self.centroids[0].mean
        if q >= 1:
            return self.centroids[-1].mean
        
        target_count = q * self.count
        
        # Binary search through cumulative counts
        for i, cumulative in enumerate(self._cumulative_counts):
            if cumulative >= target_count:
                if i == 0:
                    return self.centroids[0].mean
                
                # Linear interpolation between centroids
                prev_cum = self._cumulative_counts[i-1] if i > 0 else 0
                fraction = (target_count - prev_cum) / (cumulative - prev_cum)
                
                prev_mean = self.centroids[i-1].mean if i > 0 else self.centroids[0].mean
                return prev_mean + fraction * (self.centroids[i].mean - prev_mean)
        
        return self.centroids[-1].mean
    
    def _can_merge(self, centroid: Centroid, value: float) -> bool:
        if centroid.count == 1:
            return True
        
        # Simple merge criterion based on delta
        proposed_count = centroid.count + 1
        max_count = 4 * self.count * self.delta
        
        return proposed_count <= max_count
    
    def _compress(self):
        if len(self.centroids) <= 1:
            return
        
        # Sort centroids by mean
        self.centroids.sort(key=lambda c: c.mean)
        
        # Merge adjacent centroids that are too close
        compressed = [self.centroids[0]]
        
        for i in range(1, len(self.centroids)):
            current = self.centroids[i]
            last = compressed[-1]
            
            # Check if we can merge
            total_count = last.count + current.count
            max_count = 4 * self.count * self.delta
            
            if total_count <= max_count:
                # Merge into last centroid
                last.add(current.mean, current.count)
            else:
                compressed.append(current)
        
        self.centroids = compressed
    
    def _update_cumulative(self):
        self._cumulative_counts = []
        cumulative = 0
        for centroid in self.centroids:
            cumulative += centroid.count
            self._cumulative_counts.append(cumulative)
    
    def size(self) -> int:
        return len(self.centroids)
    
    def total_count(self) -> int:
        return self.count