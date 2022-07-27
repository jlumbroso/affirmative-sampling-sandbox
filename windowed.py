
import copy
import typing

# type: ignore
import randomhash

DEFAULT_K = 100
DEFAULT_W = 1000
DEFAULT_DEBUG = True

class DebugStatistics:

    def __init__(
        self,
        initial_value: int = 0,
        debug: bool = None,
    ):  
        self._debug = debug if debug is not None else DEFAULT_DEBUG

        if self._debug:
            self._data = dict()
            self._initial_value = initial_value
    
    @property
    def data(self) -> typing.Dict[str, int]:
        if self._debug:
            # return dict with sorted keys
            return {
                key: self._data[key]
                for key in sorted(self._data)
            }

    def get(self, key: str) -> int:
        if self._debug:
            return self._data.get(key, self._initial_value)

    def update(self, key, increment) -> int:
        if self._debug:
            self._data[key] = (
                self._data.get(key, self._initial_value) +
                increment
            )
            return self._data[key]


class AffSample:

    def __init__(
        self,
        k: int = DEFAULT_K,
        seed: typing.Optional[int] = None,
        debug: bool = None,
    ):
        self._k = k
        self._prng = randomhash.RandomHashFamily(count=1, seed=seed)

        # three data structures to store the sample
        self._sample_core = set()
        self._sample_xtra = set()
        self._sample_freqs = dict()

        # cached thresholds for the sample
        self._threshold_core_kth = None
        self._threshold_core_kth_z = None
        self._threshold_xtra_min = None
        self._threshold_xtra_min_z = None
        
        # DEBUG: statistic tracker
        self._debug = debug if debug is not None else DEFAULT_DEBUG
        self._stats = DebugStatistics()

    def _hash(self, z: str) -> int:
        return self._prng.hash(z)
    
    def _hash_preimage_pair(self, z: str) -> typing.Tuple[int, str]:
        return (self._hash(z), z)

    def _update_thresholds(self):
        # DEBUG: check invariants
        self._check_invariants()

        # compute the thresholds for the sample
        kth_hash, z_kth_hash = min(map(self._hash_preimage_pair, self._sample_core))
        min_hash, z_min_hash = (
            min(map(self._hash_preimage_pair, self._sample_xtra))

            if len(self._sample_xtra) > 0 else

            # if xtra is empty, the threshold to get in is the kth element hash
            (kth_hash, z_kth_hash)
        )

        # update internal values
        self._threshold_core_kth = kth_hash
        self._threshold_core_kth_z = z_kth_hash
        self._threshold_xtra_min = min_hash
        self._threshold_xtra_min_z = z_min_hash

    @property
    def threshold_core_kth(self) -> int:
        if self._threshold_core_kth is None:
            self._update_thresholds()
        return self._threshold_core_kth
    
    @property
    def threshold_core_kth_z(self) -> str:
        if self._threshold_core_kth_z is None:
            self._update_thresholds()
        return self._threshold_core_kth_z
    
    @property
    def threshold_xtra_min(self) -> int:
        if self._threshold_xtra_min is None:
            self._update_thresholds()
        return self._threshold_xtra_min
    
    @property
    def threshold_xtra_min_z(self) -> str:
        if self._threshold_xtra_min_z is None:
            self._update_thresholds()
        return self._threshold_xtra_min_z

    def _check_invariants(self):
        if self._debug:

            # sample core should always contain at most k elements
            assert len(self._sample_core) <= self._k

            # sample xtra should never contain any element unless sample
            # core is already filled
            assert len(self._sample_xtra) == 0 or len(self._sample_core) == self._k
            
            if len(self._sample_xtra) > 0:
                # the largest element of sample xtra should be smaller than the
                # smallest element of sample core
                assert max(map(self._hash, self._sample_xtra)) < min(map(self._hash, self._sample_core))

                # the sets should have no element in common
                assert self._sample_xtra.isdisjoint(self._sample_core)
            
            # every element of sample freqs should be in the sample
            for z in self._sample_freqs.keys():
                assert (
                    (z in self._sample_core and z not in self._sample_xtra) or
                    (z not in self._sample_core and z in self._sample_xtra)
                )
            
            # every element of sample core should be in sample freqs
            for z in self._sample_core:
                assert z in self._sample_freqs
            
            # every element of sample xtra should be in sample freqs
            for z in self._sample_xtra:
                assert z in self._sample_freqs
            
    def process(
        self,
        z: typing.Any,
    ):

        # DEBUG: update statistics
        self._stats.update('total_tokens', 1)

        # DEBUG: check invariants
        self._check_invariants()
        
        # ================================================================
        # (A1) if z is already in the sample (and therefore no need to hash it)
        if z in self._sample_core or z in self._sample_xtra:
            # update frequency
            # NOTE: this is the place we would update any other stat of z
            self._sample_freqs[z] = self._sample_freqs.get(z, 0) + 1

            # DEBUG: update statistics
            self._stats.update('case_A1', 1)
            self._stats.update('sample_tokens', 1)

            # since no new element is added, just a repetition, no other
            # update to the data structure is need -> END(added)

            return True
        
        # (A2) if the core sample is not filled, add z to the core sample
        if len(self._sample_core) < self._k:
            self._sample_core.add(z)
            self._sample_freqs[z] = 1

            # DEBUG: update statistics
            self._stats.update('case_A2', 1)
            self._stats.update('sample_tokens', 1)
            self._stats.update('sample_unique_tokens', 1)

            # new element added, update thresholds -> END(added)
            self._update_thresholds()

            return True
        
        # ================================================================
        # (B) z is not in the sample + the core sample is filled
        
        # so now, if the hash of z is not provided, compute it
        z_hash = self._prng.hash(z)
        
        # (B.1) if hash(z) is smaller than the min hash value in S
        if z_hash < self.threshold_xtra_min:
            # DISCARD = ignore the element entirely

            # DEBUG: update statistics
            self._stats.update('case_B1', 1)

            # END(dropped)
            return False

        # (B.2) if hash(z) is larger than the k-th largest hash value in S
        # hash(z) is a new k-record, needs not to be larger than the largest in the sample!
        elif z_hash > self.threshold_core_kth:
            # EXPAND = the total size sample grows by one
            # (but sample_core remains at size k)

            # add z to S
            self._sample_core.add(z)
            self._sample_freqs[z] = 1

            # move z_kth_hash from core to xtra
            self._sample_xtra.add(self._threshold_core_kth_z)
            self._sample_core.remove(self._threshold_core_kth_z)

            # DEBUG: update statistics
            self._stats.update('case_B2', 1)
            self._stats.update('sample_tokens', 1)
            self._stats.update('sample_unique_tokens', 1)

            # new element added, update thresholds -> END(added)
            self._update_thresholds()

            return True
        
        # (B.3) otherwise z replaces the element z* with min. hash value
        # sample_xtra must contain z_min_hash otherwise we have a contradiction
        else:
            # REPLACE = the size of the sample does not change but we
            # make sure to keep the largest elements

            # DEBUG: should never happen (right?)
            assert len(self._sample_xtra) > 0

            # remove the element with min. hash value
            self._sample_xtra.remove(self._threshold_xtra_min_z)
            token_count_removed = self._sample_freqs[self._threshold_xtra_min_z]
            del self._sample_freqs[self._threshold_xtra_min_z]

            # and replace by the current element z
            self._sample_xtra.add(z)
            self._sample_freqs[z] = 1

            # DEBUG: update statistics
            self._stats.update('case_B3', 1)
            self._stats.update('sample_tokens', 1 - token_count_removed)
            self._stats.update('sample_replaced_tokens', 1)

            # new element added, update thresholds -> END(added)
            self._update_thresholds()

            return True

    def remove(self, z: str) -> typing.NoReturn:
        assert z in self._sample_freqs, "z not in sample"

        # DEBUG: check invariants
        self._check_invariants()

        if z in self._sample_xtra:
            # if the element is in the xtra, remove it from there
            self._sample_xtra.remove(z)
            
        elif z in self._sample_core:
            # if the element is in the core, remove it
            self._sample_core.remove(z)

            # but since the core should not be depleted if avoidable
            # transfer the largest element of the xtra to the core
            if len(self._sample_xtra) > 0:
                max_hash, z_max_hash = max(map(self._hash_preimage_pair, self._sample_xtra))
                self._sample_xtra.remove(z_max_hash)
                self._sample_core.add(z_max_hash)

        del self._sample_freqs[z]

        # element removed, recompute thresholds
        self._update_thresholds()

    @property
    def size(self) -> int:
        return len(self._sample_core) + len(self._sample_xtra)

    @property
    def sample(self) -> typing.Dict[str, int]:
        return copy.deepcopy(self._sample_freqs)

    @property
    def statistics(self) -> typing.Dict[str, int]:
        return self._stats.data

    def contains(self, z: str) -> bool:
        return z in self._sample_freqs
    
    @property
    def cardinality_estimate(self) -> int:
        # KMV formula
        return (len(self._sample_freqs)-1)/(1-randomhash.int_to_real(self._threshold_xtra_min))

        


class WindowedV1AffSample(AffSample):

    def __init__(
        self,
        k: int = DEFAULT_K,
        w: int = DEFAULT_W,
        seed: typing.Optional[int] = None,
        debug: bool = None,
    ):
        super().__init__(k=k, seed=seed, debug=debug)

        # windowed stuff
        self._time = 0
        self._w = w
        self._latest_timestamp = dict()

    def remove(self, z: str):
        super().remove(z)
        del self._latest_timestamp[z]

    def _clear_outdated(self):
        
        # making copy so it can be updated in the loop
        elements_to_consider = list(self._latest_timestamp.items())[:]

        for z, timestamp in elements_to_consider:

            if timestamp < self._time - self._w and self.contains(z):
                self.remove(z)

    def process(
        self,
        z: typing.Any,
    ):
        self._clear_outdated()
        is_added = super().process(z=z)
        if is_added:  
            self._latest_timestamp[z] = self._time
        self._time += 1
    
    @property
    def w(self):
        return self._w


class WindowedV2AffSample:

    def __init__(
        self,
        k: int = DEFAULT_K,
        w: int = DEFAULT_W,
        debug: bool = None,
    ):
        self._k = k
        self._w = w

        # DEBUG: statistic tracker
        self._debug = debug if debug is not None else DEFAULT_DEBUG
        self._stats = DebugStatistics()

        self._main_sample = AffSample(k=self._k, debug=self._debug)
        self._auxi_sample = AffSample(k=self._k, debug=self._debug)
        self._time = 0
    
    def process(
        self,
        z: str,
    ):  
        if self._time % self._w == 0:
            # swap main and auxi samples
            self._main_sample = self._auxi_sample
            self._auxi_sample = AffSample(k=self._k, debug=self._debug)

        self._main_sample.process(z=z)
        self._auxi_sample.process(z=z)

        self._time += 1
    
    @property
    def sample(self):
        return self._main_sample.sample
    
    @property
    def statistics(self) -> typing.Dict[str, int]:
        data = dict()
        data.update({
            "{}.{}".format("main", key): value
            for key, value in self._main_sample.statistics.items()
        })
        data.update({
            "{}.{}".format("auxi", key): value
            for key, value in self._auxi_sample.statistics.items()
        })
        return data

    @property
    def w(self):
        return self._w

    @property
    def size(self) -> int:
        return len(self.sample)

    def contains(self, z: str) -> bool:
        return z in self.sample
    
    @property
    def cardinality_estimate(self) -> int:
        return self._main_sample.cardinality_estimate
    