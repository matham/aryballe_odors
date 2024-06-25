import csv
from sklearn.naive_bayes import GaussianNB
from pathlib import Path
import re
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.decomposition import PCA
import math
from itertools import combinations
from functools import partial
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable


HEADER_PAT = re.compile(r"([0-9]+)\[([A-Z]+[0-9]+)]")

PRISM_TRIAL_START_PAT = re.compile(r"Valve in Injection position \((.+)\)")
PRISM_TRIAL_ITI_NAME = "Rinse"

REC_NUM = 0
RELATIVE_TIME = 1
UNIX_TIME = 2
RUN_ID = 3
RUN_NAME = 4
DEVICE_ID = 5
ITEM_NAME = 6
CYCLE_NUM = 7
SECTION_NAME = 8
HUMIDITY_VAL = 9
TEMP_VAL = 10
SENSOR0 = 11

BLANK_SENSOR_ARYBALLE = "1"

BLANK_SENSOR_PRISM = "Au"


class TrialSegment:
    """Data of component of a trial.

    A trial is composed of e.g. a baseline measurement, analyte and, post analyte measurement. This is the data
    of a single segment, e.g. the baseline part of the trial.
    """

    sensor_id: list[str]
    """A 1d list of the same number of columns of `data`, each item is the id of the peptide. A peptide can be
    represented multiple times due to replicates.
    """

    sensor_id_unique: list[int | str]
    """A 1d list of the same number of columns of `data`, each item is the position info of the peptide and unique.
    """

    time_running: np.ndarray
    """A 1d of same length as rows in `data`. Each item is the time in seconds corresponding to the row.

    This time value resets for each trial, but is continuous across the baseline, analyte, and post analyte segments.
    So the first value may be non-zero.
    """

    time: np.ndarray
    """Same as `time_running`, except we subtract the first element so it starts with time zero.
    """

    unix_time: np.ndarray
    """A 1d of same length as rows in `data`. Each item is the unix time corresponding to the row.
    """

    humidity: np.ndarray
    """A 1d of same length as rows in `data`. Each item is humidity sensor measurement corresponding to the row.
    """

    temp: np.ndarray
    """A 1d of same length as rows in `data`. Each item is temperature sensor measurement corresponding to the row.
    """

    data: np.ndarray
    """A 2d array. Rows is time, columns is sensors (peptides).
    """

    desorption_linear_offset: np.ndarray | None = None

    def __init__(
            self, sensor_id, sensor_id_unique, relative_time, unix_time, humidity, temp, data):
        self.sensor_id = sensor_id
        self.sensor_id_unique = sensor_id_unique
        self.time_running = relative_time
        self.time = relative_time - relative_time[0]
        self.duration = self.time[-1].item()
        self.unix_time = unix_time
        self.humidity = humidity
        self.temp = temp
        self.data = data

    def segment_from_data(self, data, sensor_id=None, sensor_id_unique=None) -> "TrialSegment":
        cls = self.__class__
        if sensor_id is None:
            sensor_id = self.sensor_id
        if sensor_id_unique is None:
            sensor_id_unique = self.sensor_id_unique

        if data.shape[-1] != len(sensor_id):
            raise ValueError("Data size doesn't match sensor ids")

        seg = cls(
            sensor_id=sensor_id, sensor_id_unique=sensor_id_unique, relative_time=self.time_running,
            unix_time=self.unix_time, humidity=self.humidity, temp=self.temp, data=data
        )
        return seg


class AryballeRecord:
    """Data of a single trial.

    A trial is composed of a baseline measurement, analyte measurement, and post analyte measurement. This contains
    the data for all 3 components.
    """

    record: int
    """The number of this record within the sequence of trials, assigned by Aryballe. It is monotonically increasing
    with each trial sequentially (seemingly).
    """

    run_id: str
    """The ID assigned by Aryballe to this record.
    """

    run_name: str
    """The name assigned by Aryballe to this record.
    """

    device_id: str
    """The device ID used to collect this data.
    """

    sample_name: str
    """The name of the analyte used as the odor in this trial.
    """

    cycle: int
    """The count for this analyte.

    Aryballe will iterate through each analyte (odor) a specified number of times, interleaving analytes. This is the
    number of cycles this analyte has been repeated so far, starting from 1.
    """

    sensor_id: list[str]
    """Same as TrialSegment.sensor_id.
    """

    sensor_id_unique: list[int | str]
    """Same as TrialSegment.sensor_id_unique.
    """

    baseline: TrialSegment
    """The data for the baseline segment of the trial.
    """

    odor: TrialSegment
    """The data for the analyte (odor) segment of the trial.

    This follows immediately after the data of `baseline` so the data arrays can be concatenated to get the overall
    trial data. See `TrialSegment.running_time`.
    """

    iti: TrialSegment
    """The data for the post analyte segment of the trial.

    This follows immediately after the data of `odor` and `baseline` so the data arrays can be concatenated to get the
    overall trial data. See `TrialSegment.running_time`.
    """

    tags: set[str]
    """A set of tags associated with the record.
    """

    h2o_affinity: np.ndarray | None = None

    blank_sensor: str

    def __init__(
            self, record, run_id, run_name, device_id, sample_name,
            cycle, sensor_id, sensor_id_unique, baseline, odor, iti, tags, h2o_affinity, blank_sensor):
        self.record = record
        self.run_id = run_id
        self.run_name = run_name
        self.device_id = device_id
        self.sample_name = sample_name
        self.cycle = cycle
        self.sensor_id = sensor_id
        self.sensor_id_unique = sensor_id_unique
        self.baseline = baseline
        self.odor = odor
        self.iti = iti
        self.tags = tags
        self.h2o_affinity = h2o_affinity
        self.blank_sensor = blank_sensor

    def __str__(self):
        return (
            f"<{id(self)}@AryballeRecord sample=\"{self.sample_name}\", cycle={self.cycle}, "
            f"times=({self.baseline.duration:0.2f},{self.odor.duration:0.2f},{self.iti.duration:0.2f}), "
            f"tags=[{','.join(sorted(self.tags))}]>"
        )

    def record_from_normalization(self, normalization: set) -> "AryballeRecord":
        record = self.__class__(
            self.record, run_id=self.run_id, run_name=self.run_name, device_id=self.device_id,
            sample_name=self.sample_name, cycle=self.cycle, sensor_id=[], sensor_id_unique=[], baseline=None, odor=None,
            iti=None, tags=self.tags, h2o_affinity=self.h2o_affinity, blank_sensor=self.blank_sensor,
        )

        seg: TrialSegment
        for name in ("baseline", "odor", "iti"):
            seg = getattr(self, name)
            if seg is None:
                continue

            normalized, _, sensor_ids = self.get_data(name, normalization)
            new_seg = seg.segment_from_data(normalized, sensor_ids, sensor_ids)

            setattr(record, name, new_seg)
            record.sensor_id = sensor_ids
            record.sensor_id_unique = sensor_ids

        return record

    def record_from_transform(self, transformer, labels: list[str], labels_unique: list[str] | None = None) -> "AryballeRecord":
        if labels_unique is None:
            labels_unique = labels
        record = self.__class__(
            self.record, run_id=self.run_id, run_name=self.run_name, device_id=self.device_id,
            sample_name=self.sample_name, cycle=self.cycle, sensor_id=labels, sensor_id_unique=labels_unique,
            baseline=None, odor=None, iti=None, tags=self.tags, h2o_affinity=self.h2o_affinity,
            blank_sensor=self.blank_sensor,
        )

        seg: TrialSegment
        for name in ("baseline", "odor", "iti"):
            seg = getattr(self, name)
            if seg is None:
                continue

            new_data = transformer(seg.data)
            new_seg = seg.segment_from_data(new_data, labels, labels_unique)

            setattr(record, name, new_seg)

        return record

    def compute_next_record_slope_offset(
            self, next_record: "AryballeRecord", end_fraction: float = 1 / 100, record_dt_threshold: float = 30
    ):
        if self.iti is None or next_record.baseline.unix_time[0] - self.iti.unix_time[-1] > record_dt_threshold:
            next_record.baseline.desorption_linear_offset = None
            next_record.odor.desorption_linear_offset = None
            if next_record.iti is not None:
                next_record.iti.desorption_linear_offset = None

            return

        n_sensors = self.odor.data.shape[1]
        next_baseline_t = next_record.baseline.unix_time
        next_baseline_offset = np.zeros((len(next_baseline_t), n_sensors))
        next_odor_t = next_record.odor.unix_time
        next_odor_offset = np.zeros((len(next_odor_t), n_sensors))
        next_iti_t = None if next_record.iti is None else next_record.iti.unix_time
        next_iti_offset = None if next_iti_t is None else np.zeros((len(next_iti_t), n_sensors))

        n_samples = int(len(self.iti.unix_time) * (1 - end_fraction))
        data = self.iti.data[n_samples:, :]
        times = self.iti.unix_time[n_samples:]
        for i in range(data.shape[1]):
            poly = np.polynomial.Polynomial.fit(times, data[:, i], deg=1)
            # r_sq = 1.0 - (np.var(poly(times) - data[:, i]) / np.var(data[:, i]))

            next_baseline_offset[:, i] = poly(next_baseline_t)
            next_odor_offset[:, i] = poly(next_odor_t)
            if next_iti_t is not None:
                next_iti_offset[:, i] = poly(next_iti_t)

        next_record.baseline.desorption_linear_offset = next_baseline_offset
        next_record.odor.desorption_linear_offset = next_odor_offset
        if next_iti_t is not None:
            next_record.iti.desorption_linear_offset = next_iti_offset

    def compute_water_h2o_affinity(self):
        return (
            (self.odor.data[-1, :] - self.baseline.data[-1, :]) /
            (self.odor.humidity[-1] - self.baseline.humidity[-1])
        )

    def _get_sections(self, section: str | None) -> list[TrialSegment]:
        match section:
            case None | "baseline+odor+iti":
                # all sections
                return [self.baseline, self.odor, self.iti]
            case "baseline+odor":
                return [self.baseline, self.odor]
            case "odor+iti":
                return [self.odor, self.iti]
            case "baseline":
                return [self.baseline]
            case "odor":
                return [self.odor]
            case "iti":
                return [self.iti]
            case _:
                raise ValueError(f"Unknown sections {section}")

    def get_data(
            self, section, normalization: set, reset_time=True
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Gets the 2d data (same axis as `TrialSegment.data`) and associated timestamps of the record.

        :param section: The data components - it concatenates the requested data. Can be None, which returns the same as
            `"baseline+odor+iti"`. Also accepts, `"baseline+odor"`, `"odor+iti"`, `"baseline"`, `"odor"`, and `"iti"`
            that returns a single array of the data.
        :param normalization: See `normalize_data`.
        :param reset_time: Whether to reset the timestamp of the first data row to zero.
        :return: A tuple of data and time. Data is time X sensors (rows, columns). Time is the 1d array of the
        associated continuous timestamps of the data.
        """
        items = self._get_sections(section)

        data = np.concatenate([getattr(item, "data") for item in items], axis=0)

        humidity_items = [getattr(item, "humidity") for item in items if getattr(item, "humidity") is not None]
        humidity = None
        if humidity_items:
            humidity = np.concatenate(humidity_items, axis=0)

        desorption_linear_offset = None
        if self.baseline.desorption_linear_offset is not None:
            desorption_linear_offset = np.concatenate(
                [getattr(item, "desorption_linear_offset") for item in items], axis=0)
        time = np.concatenate([getattr(item, "time_running") for item in items], axis=0).copy()
        if reset_time:
            time -= time[0]

        data, sensor_ids = self.normalize_data(
            data, normalization=normalization, humidity=humidity,
            desorption_linear_offset=desorption_linear_offset
        )

        return data, time, sensor_ids

    def get_humidity_data(self, section, reset_time=True) -> tuple[np.ndarray, np.ndarray]:
        """Gets the ---------- and associated timestamps of the record.

        :param section: The data components - it concatenates the requested data. Can be None, which returns the same as
            `"baseline+odor+iti"`. Also accepts, `"baseline+odor"`, `"odor+iti"`, `"baseline"`, `"odor"`, and `"iti"`
            that returns a single array of the data.
        :param reset_time: Whether to reset the timestamp of the first data row to zero.
        :return: A tuple of data and time. Data is time X sensors (rows, columns). Time is the 1d array of the
        associated continuous timestamps of the data.
        """
        items = self._get_sections(section)

        humidity = np.concatenate([getattr(item, "humidity") for item in items], axis=0)
        time = np.concatenate([getattr(item, "time_running") for item in items], axis=0).copy()
        if reset_time:
            time -= time[0]

        return humidity, time

    def dedup_sensor_data(self, data: np.ndarray) -> tuple[list[str], np.ndarray]:
        out_data = []
        sensor_ids = []
        for sensor, mask in sorted(self._get_sensor_masks().items(), key=lambda x: x[0]):
            out_data.append(np.median(data[:, mask], axis=1, keepdims=True))
            sensor_ids.append(sensor)

        out_data = np.concatenate(out_data, axis=1)
        return sensor_ids, out_data

    def _get_sensor_masks(self, sensor_ids=None) -> dict[str, np.ndarray]:
        """Gets a dict mapping sensor IDs to a mask of the columns of the repeats of this sensor (peptide).

        Each key is the ID of the sensor (e.g. 25). The corresponding value is the logical ndarray which has true values
        at the indices (columns) corresponding to the repeats of this peptide. Indexing `data` or `sensor_id_unique` will
        return the columns (values) of this sensor repeats.
        """
        if sensor_ids is None:
            sensor_ids = self.sensor_id

        masks = {}
        for sensor_id in set(sensor_ids):
            masks[sensor_id] = np.array([val == sensor_id for val in sensor_ids], dtype=np.bool_)

        return masks

    def get_baseline_offset(
            self, keep2dims=True, subtract_humidity=False, end_duration=1, baseline_humidity_index=-1
    ) -> np.ndarray:
        """Returns the offset of each sensor to zero it.

        :param keep2dims: If True will keep the first dim to have a shape of 1 X N, otherwise just 1D N sized array.
        :return: An array, that is the offset for each sensor. Such that subtracting it from the corresponding sensor
            will keep the average value at zero, as computed from the baseline data.
        """
        data = self.baseline.data.copy()
        if subtract_humidity:
            data -= self._get_sensor_humidity_offset(
                self.baseline.humidity, baseline_index=baseline_humidity_index, keep2dims=True)

        time = self.baseline.time
        mask = time >= time[-1] - end_duration
        return np.median(data[mask, :], axis=0, keepdims=keep2dims)

    def _get_sensor_humidity_offset(self, humidity: np.ndarray | float, baseline_index=-1, keep2dims=True):
        humidity = np.atleast_1d(humidity)
        if len(humidity.shape) == 1:
            humidity = humidity[:, np.newaxis]

        h2o_affinity = self.h2o_affinity[np.newaxis, :]
        offset = h2o_affinity * (humidity - self.baseline.humidity[baseline_index])

        if keep2dims:
            return offset
        return offset.squeeze()

    def normalize_data(
            self, data, normalization: set, humidity: np.ndarray | float,
            desorption_linear_offset: np.ndarray | None | float
    ) -> tuple[np.ndarray, list[str]]:
        """Normalizes the input data.

        Normalization is computed by removing the offset computed from the baseline. Then, it's scaled by the magnitude
        at the end of the analyte period. It is computed as a single value for all the sensors.
        """
        data = data.copy()
        sensor_ids = self.sensor_id
        d1 = len(data.shape) == 1
        blank_idx = np.array([id_ == self.blank_sensor for id_ in sensor_ids], dtype=np.bool_)
        if d1:
            data = data[np.newaxis, :]

        if "humidity" in normalization:
            idx = -1 if "baseline" in normalization else 0
            data -= self._get_sensor_humidity_offset(humidity, baseline_index=idx)

        if "baseline" in normalization:
            offset = self.get_baseline_offset(keep2dims=True, subtract_humidity=False)
            data -= offset

        if "desorption" in normalization and desorption_linear_offset is not None:
            offset = self.baseline.desorption_linear_offset[-1, :][np.newaxis, :]
            desorption_linear_offset = desorption_linear_offset - offset
            data -= desorption_linear_offset

        if "blank" in normalization:
            data -= np.median(data[:, blank_idx], axis=1, keepdims=True)

        if "dedup" in normalization:
            sensor_ids, data = self.dedup_sensor_data(data)

        if "norm_to_area" in normalization or "norm_to_area_peak" in normalization:
            norm_data = data
            if "norm_to_area_peak" in normalization:
                if "dedup" in normalization:
                    _, norm_data = self.dedup_sensor_data(self.odor.data[-1, :][np.newaxis, :])
                else:
                    norm_data = self.odor.data[-1, :][np.newaxis, :]

            # medians = []
            # for mask in self._get_sensor_masks().values():
            #     medians.append(np.median(norm_data[:, mask], axis=1, keepdims=True))
            # scale = np.linalg.norm(np.hstack(medians), axis=1, keepdims=True)
            # scale = np.sum(np.hstack(np.abs(medians)), axis=1, keepdims=True)
            scale = np.sum(np.abs(norm_data), axis=1, keepdims=True)

            data = np.divide(data, scale, out=np.zeros_like(data), where=scale != 0)

        if "norm_range" in normalization:
            data -= np.min(data, axis=1, keepdims=True)
            max_val = np.max(data, axis=1, keepdims=True)
            data = np.divide(data, max_val, out=np.zeros_like(data), where=max_val != 0)

        if d1:
            return data[0, :], sensor_ids
        return data, sensor_ids

    def get_data_by_sensor(
            self, normalization: set, section=None, reset_time=True
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Returns the sensor data, split by sensor (peptide) type.

        :param section: See `get_data`.
        :param normalization: See `normalize_data`.
        :param reset_time: See `get_data`.
        :return: Similar to `get_data`, but split into mapping for sensor (peptide) IDs to repeats of the peptide.
            I.e. keys are the IDs of the peptides. Each value is a 2d array, with rows being time, and columns are
            the repeats of only this peptide.
        """
        data, time, sensor_ids = self.get_data(
            section, normalization=normalization, reset_time=reset_time)

        sensor_data = {}
        for sensor_id, mask in self._get_sensor_masks(sensor_ids).items():
            sensor_data[sensor_id] = data[:, mask]

        return sensor_data, time

    def get_final_odor_sensor_data(self, normalization: set) -> dict[str, np.ndarray]:
        """Returns the sensor data from the end of the analyte (odor segment), split by sensor (peptide) type.

        :param normalization: See `normalize_data`.
        :return: The last data sample of the odor segment, split by peptide. I.e. a mapping of sensor (peptide) IDs to
            repeats of the peptide. Keys are the IDs of the peptides. Each value is a 1d array with the value of the
            repeats of only this peptide.
        """
        data, sensor_ids = self.get_flat_final_odor_sensor_data(normalization=normalization)

        sensor_data = {}
        for sensor_id, mask in self._get_sensor_masks(sensor_ids).items():
            sensor_data[sensor_id] = data[mask]

        return sensor_data

    def get_flat_final_odor_sensor_data(self, normalization: set) -> tuple[np.ndarray, list[str]]:
        """Returns the sensor data from the end of the analyte (odor segment).

        :param normalization: See `normalize_data`.
        :return: The last data sample of the odor segment as 1d array, where each value corresponds to a sensor
            in `sensor_id`.
        """
        desorption_linear_offset = None
        if self.odor.desorption_linear_offset is not None:
            desorption_linear_offset = self.odor.desorption_linear_offset[-1]

        humidity = None
        if self.odor.humidity is not None:
            humidity = self.odor.humidity[-1]
        data, sensor_ids = self.normalize_data(
            self.odor.data[-1, :], normalization=normalization, humidity=humidity,
            desorption_linear_offset=desorption_linear_offset
        )

        return data, sensor_ids

    def estimate_baseline_slope(self) -> dict[str, np.ndarray]:
        """Estimates the slope of the time series baseline data.

        Returns a dict whose keys is the sensor (peptide) ID and values is the estimate of the time varying slope
        for that sensor ID. It's the average of all the repeats for each sensor.
        """
        data = self.baseline.data

        slopes = {}
        for sensor_id, mask in self._get_sensor_masks().items():
            res = stats.linregress(self.baseline.time, np.median(data[:, mask], axis=1))
            slopes[sensor_id] = res.slope

        return slopes

    def estimate_iti_slope(self, end_duration=None) -> dict[str, np.ndarray]:
        """Estimates the slope of the time series post-analyte (odor) data as the data goes back to baseline.

        :param end_duration: If not None, the number of seconds from the end of the segment to use.
        :return: A dict whose keys is the sensor (peptide) ID and values is the estimate of the time varying slope
            for that sensor ID. It's the average of all the repeats for each sensor.
        """
        data = self.iti.data
        time = self.iti.time

        if end_duration is not None:
            t = self.iti.time[-1] - end_duration
            n = np.sum(self.iti.time < t)
            time = time[n:]
            data = data[n:, :]

        slopes = {}
        for sensor_id, mask in self._get_sensor_masks().items():
            res = stats.linregress(time, np.median(data[:, mask], axis=1))
            slopes[sensor_id] = res.slope

        return slopes


class AryballeRecords:
    """Parses and visualizes the Aryballe sensor records.
    """

    records: list[AryballeRecord]

    h2o_affinity: np.ndarray | None = None

    @property
    def sensor_id(self) -> list[str]:
        """See TrialSegment.sensor_id.
        """
        return self.records[0].sensor_id

    @property
    def sensor_id_unique(self) -> list[int | str]:
        """See TrialSegment.sensor_id_unique.
        """
        return self.records[0].sensor_id_unique

    def __init__(self):
        self.records = []

    def process_records(self, records: list[AryballeRecord], normalization: set) -> None:
        self.records = [r.record_from_normalization(normalization) for r in records]

    def add_aryballe_csv_records(self, *filenames: str) -> None:
        """Adds the records from the given CSV file to `records`.

        :param filename: The CSV filename as exported from Aryballe hub.
        """
        for filename in filenames:
            self.records.extend(self.parse_records_aryballe_csv(filename))

    def _split_rows_by_id(self, data: list, idx: int) -> list[tuple[int, int]]:
        """Splits the 2d `data` into groups of rows, using the column `idx` such that each group of rows has all the
        same value for the column.

        :return: A list of tuples, where each tuple is the start and end index in `data` for a consecutive group of
            rows. The end index is the index of the row *after* the last row of that group.
        """
        last = None
        indices = []

        for i, row in enumerate(data):
            if row[idx] != last:
                indices.append(i)
                last = row[idx]

        return list(zip(indices, indices[1:] + [len(data)]))

    def parse_records_aryballe_csv(self, filename: str) -> list[AryballeRecord]:
        """Parses and returns the records from the given CSV file.

        :param filename: The CSV filename as exported from Aryballe hub.
        """
        with open(filename, 'r') as fh:
            reader = list(csv.reader(fh, delimiter=","))

        header = reader[0]

        n_sensors = 0
        sensor_id = []
        sensor_id_unique = []
        for item in header[SENSOR0:]:
            m = re.match(HEADER_PAT, item)
            if m is not None:
                n_sensors += 1
                id_, c = m.groups()
                sensor_id.append(id_)
                sensor_id_unique.append(c)
            else:
                break

        tags = []
        for i, item in enumerate(header[SENSOR0 + n_sensors:], SENSOR0 + n_sensors):
            if not item:
                break
            tags.append((i, item))

        rows = reader[1:]

        records = []
        for s, e in self._split_rows_by_id(rows, REC_NUM):
            record_data = rows[s: e]
            row0 = record_data[0]

            segments = self._split_rows_by_id(record_data, SECTION_NAME)
            assert len(segments) in (2, 3), "expected 2 or 3 segments in trial"

            trials = []
            for rec_s, rec_e in segments:
                segment_data = record_data[rec_s: rec_e]
                rel_time = np.array([row[RELATIVE_TIME] for row in segment_data], dtype=np.float32)
                unix_time = np.array([row[UNIX_TIME] for row in segment_data], dtype=np.float64)
                humidity = np.array([row[HUMIDITY_VAL] for row in segment_data], dtype=np.float32)
                temp = np.array([row[TEMP_VAL] for row in segment_data], dtype=np.float32)
                data = np.array([row[SENSOR0: SENSOR0 + n_sensors] for row in segment_data], dtype=np.float32)

                trial = TrialSegment(
                    sensor_id=sensor_id, sensor_id_unique=sensor_id_unique, relative_time=rel_time, unix_time=unix_time,
                    humidity=humidity, temp=temp, data=data
                )
                trials.append(trial)

            if len(trials) == 2:
                trials.append(None)

            baseline, odor, iti = trials

            record_tags = set()
            for i, name in tags:
                if row0[i] == 'true':
                    record_tags.add(name)

            record = AryballeRecord(
                record=int(row0[REC_NUM]), run_id=int(row0[RUN_ID]), run_name=row0[RUN_NAME], device_id=row0[DEVICE_ID],
                sample_name=row0[ITEM_NAME].replace("-", " "), cycle=int(row0[CYCLE_NUM]), sensor_id=sensor_id, sensor_id_unique=sensor_id_unique,
                baseline=baseline, odor=odor, iti=iti, tags=record_tags, h2o_affinity=self.h2o_affinity, blank_sensor=BLANK_SENSOR_ARYBALLE,
            )

            records.append(record)

        return records

    def add_prism_xlsx_records(self, *filenames: str) -> None:
        for filename in filenames:
            self.records.extend(self.parse_records_prism_xlsx(filename))

    def _correct_prism_name(self, name: str) -> str:
        m = re.match(r"([0-9]+%) +(.+)", name)
        if m is None:
            return name

        p, label = m.groups()
        return f"{label} {p}"

    def _parse_prism_experiment_structure(self, filename: str) -> list[tuple[str, float, float, float | None]]:
        plan = pd.read_excel(
            filename, sheet_name="Comments", skiprows=2, header=None,
            usecols=[0, 1, 2, 3, 4],
        )

        trials = []

        name = None
        trial_s = None
        iti_s = None
        for row in range(len(plan)):
            label = plan[4][row]
            t = plan[3][row]

            m = re.match(PRISM_TRIAL_START_PAT, label)
            if m is not None:
                if name is not None:
                    # didn't save last trial
                    if iti_s is None:
                        trials.append((name, trial_s, t, None))
                    else:
                        trials.append((name, trial_s, iti_s, t))

                name, = m.groups()
                trial_s = t
                iti_s = None

            elif label == PRISM_TRIAL_ITI_NAME:
                if name is None:
                    continue
                iti_s = t

            else:
                if name is not None:
                    # save last trial
                    if iti_s is None:
                        trials.append((name, trial_s, t, None))
                    else:
                        trials.append((name, trial_s, iti_s, t))

                    name = trial_s = iti_s = None

        for name, trial_s, iti_s, e in trials:
            if e is None:
                print(f"Didn't find iti for {filename} trial {name}")

        return trials

    def parse_records_prism_xlsx(self, filename: str) -> list[AryballeRecord]:
        kinetics = pd.read_excel(
            filename, sheet_name="Kinetics", header=(0, 1), dtype=np.float_
        )

        n_sensors = 0
        sensor_id = []
        sensor_id_unique = []
        for item in kinetics.columns[1:]:
            n_sensors += 1
            sensor_id.append(str(item[1]))
            sensor_id_unique.append(str(item[0]))

        times = kinetics[kinetics.columns[0]].to_numpy()
        trial_num = defaultdict(int)

        records = []
        experiments = self._parse_prism_experiment_structure(filename)
        for name, trial_s, iti_s, e in sorted(experiments, key=lambda x: x[0]):
            name = self._correct_prism_name(name)

            baseline_s_sample = np.sum(times < (trial_s - .2))
            trial_s_sample = np.sum(times < trial_s)

            data = kinetics[baseline_s_sample: trial_s_sample].to_numpy()[:, 1:]
            unix_time = kinetics[baseline_s_sample: trial_s_sample].to_numpy()[:, 0]
            rel_time = unix_time - unix_time[0]
            baseline = TrialSegment(
                sensor_id=sensor_id, sensor_id_unique=sensor_id_unique, relative_time=rel_time, unix_time=unix_time,
                humidity=None, temp=None, data=data
            )

            iti_s_sample = np.sum(times < iti_s)

            data = kinetics[trial_s_sample: iti_s_sample].to_numpy()[:, 1:]
            unix_time = kinetics[trial_s_sample: iti_s_sample].to_numpy()[:, 0]
            rel_time = unix_time - unix_time[0]
            odor = TrialSegment(
                sensor_id=sensor_id, sensor_id_unique=sensor_id_unique, relative_time=rel_time, unix_time=unix_time,
                humidity=None, temp=None, data=data
            )

            iti = None
            if e is not None:
                e_sample = np.sum(times < e)

                data = kinetics[iti_s_sample: e_sample].to_numpy()[:, 1:]
                unix_time = kinetics[iti_s_sample: e_sample].to_numpy()[:, 0]
                rel_time = unix_time - unix_time[0]
                iti = TrialSegment(
                    sensor_id=sensor_id, sensor_id_unique=sensor_id_unique, relative_time=rel_time, unix_time=unix_time,
                    humidity=None, temp=None, data=data
                )

            trial_num[name] += 1
            record = AryballeRecord(
                record=trial_num[name], run_id=0, run_name=os.path.basename(filename), device_id="Prism 5",
                sample_name=name, cycle=trial_num[name] + 1, sensor_id=sensor_id,
                sensor_id_unique=sensor_id_unique,
                baseline=baseline, odor=odor, iti=iti, tags=set(), h2o_affinity=None, blank_sensor=BLANK_SENSOR_PRISM
            )

            records.append(record)

        return records

    def _get_fig_by_sensor(
            self, n_sensors, n_rows, sharey, sharex, with_humidity=False, remaining_id="None"
    ) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        n_cols = int(math.ceil((n_sensors + int(with_humidity)) / n_rows))
        fig, axs = plt.subplots(n_rows, n_cols, sharey=sharey, sharex=sharex)
        ax_flat = axs.flatten().tolist()

        # dedup in original order
        sensor_id = []
        for id_ in self.sensor_id:
            if id_ not in sensor_id:
                sensor_id.append(id_)

        ax_map: dict[str, plt.Axes] = {i: ax_flat.pop(0) for i in sensor_id}
        if with_humidity:
            ax_map["humidity"] = ax_flat.pop(0)
        ax_map[remaining_id] = ax_flat

        return fig, ax_map

    def _get_axes_for_odors(self, n_rows: int, sharex, sharey):
        odors = list(sorted(set(r.sample_name for r in self.records)))

        n_cols = int(math.ceil(len(odors) / n_rows))
        fig, axs = plt.subplots(n_rows, n_cols, sharey=False, sharex=True)
        ax_flat = axs.flatten().tolist()

        return fig, ax_flat, odors

    def _save_or_show(self, normalization_names: set, save_fig_root: str | None = None, save_fig_prefix: str = ""):
        if save_fig_root:
            Path(save_fig_root).mkdir(parents=True, exist_ok=True)

            norm = "none"
            if normalization_names:
                norm = ",".join(sorted(normalization_names))

            fig = plt.gcf()
            fig.set_size_inches(40, 16)
            fig.tight_layout()
            if fig.legends:
                legend = fig.legends[-1]
                fig_size = fig.get_size_inches()[0] * fig.dpi
                fig.subplots_adjust(right=1 - legend.get_window_extent().width / fig_size)
            fig.savefig(
                f"{save_fig_root}/{save_fig_prefix}_norm={norm}.png", bbox_inches='tight',
                dpi=300
            )
            plt.close()
        else:
            plt.tight_layout()
            fig = plt.gcf()
            if fig.legends:
                legend = fig.legends[-1]
                fig_size = fig.get_size_inches()[0] * fig.dpi
                fig.subplots_adjust(right=1 - legend.get_window_extent().width / fig_size)
            plt.show()

    def plot_by_sensor_id(
            self, n_sensors, odor_name: str, normalization_names: set, section=None, n_rows=3,
            save_fig_root: str | None = None, save_fig_prefix: str = "", show_humidity=False
    ):
        fig, ax_map = self._get_fig_by_sensor(n_sensors, n_rows, sharey=False, sharex=True, with_humidity=show_humidity)
        ax_count = defaultdict(int)

        for record in self.records:
            if record.sample_name != odor_name:
                continue

            data, time = record.get_data_by_sensor(
                normalization=set(), section=section, reset_time=True)
            for sensor, values in data.items():
                ax_map[sensor].plot(time, values, '-')
                ax_count[sensor] += values.shape[1]

            if show_humidity:
                humidity, time = record.get_humidity_data(section=section, reset_time=True)
                ax_map["humidity"].plot(time, humidity, '-')
                ax_count["humidity"] += 1

        for sensor, ax in ax_map.items():
            ax.set_title(f"Sensor {sensor} ({ax_count[sensor]} reps)")
        fig.supxlabel("Time (s)")
        fig.supylabel("Amplitude")
        fig.suptitle(odor_name)

        self._save_or_show(normalization_names, save_fig_root, save_fig_prefix + f"_{odor_name}")

    def _get_final_odor_data(
            self, odor_name: str,
    ) -> tuple[list[tuple[int, str]], list[tuple[int, np.ndarray]]]:
        sensors = []
        sensor_names = []
        sensor_id = self.sensor_id
        for record in self.records:
            if record.sample_name != odor_name:
                continue

            data = record.get_final_odor_sensor_data(normalization=set())
            sensor_data = []
            sensor_names = []
            count = 0
            for sensor, values in sorted(data.items(), key=lambda x: sensor_id.index(x[0])):
                sensor_data.append(values)
                sensor_names.append((count, f"{sensor}"))
                count += len(values)

            sensors.append((record.cycle, np.concatenate(sensor_data)))

        return sensor_names, sensors

    def _get_final_odor_data_by_odor(self, sensor_id: str | None) -> tuple[list[str], list[np.ndarray]]:
        sample_data = defaultdict(list)
        for record in self.records:
            if sensor_id is None:
                data, _ = record.get_flat_final_odor_sensor_data(normalization=set())
            else:
                data = record.get_final_odor_sensor_data(normalization=set())[sensor_id]
            sample_data[record.sample_name].append(data)

        sample_names = list(sorted(sample_data))
        sample_arrays = [np.vstack(sample_data[name]) for name in sample_names]

        return sample_names, sample_arrays

    def plot_final_odor_data_waterfall_all_odors(
            self, normalization_names: set, average_trials: bool, save_fig_root: str | None = None,
            save_fig_prefix: str = ""
    ):
        fig, ax = plt.subplots(1, 1)
        odors = list(sorted(set(r.sample_name for r in self.records)))

        all_data = []
        sensor_names = []
        odor_count = 0
        odor_ticks = []
        for odor in odors:
            sensor_names, trials = self._get_final_odor_data(odor)
            trials = sorted(trials, key=lambda x: x[0])
            data = np.asarray([t[1] for t in trials])
            if average_trials:
                data = np.median(data, axis=0, keepdims=True)

            all_data.append(data)
            odor_ticks.append((odor_count, len(odor_ticks)))
            odor_count += len(data)

        all_data = np.vstack(all_data)
        image = ax.imshow(all_data.T, origin="lower", interpolation="none", aspect="auto")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(image, cax=cax, orientation='vertical')

        ax.set_yticks([v[0] for v in sensor_names], [v[1] for v in sensor_names])
        ax.set_xticks([v[0] for v in odor_ticks], [v[1] for v in odor_ticks])
        ax.set_xlabel("Odors")
        ax.set_ylabel("Sensors")
        ax.set_title("Sensor intensity per trial")
        fig.legend(
            handles=[mpatches.Patch(label=f"{i} - {n}", fill=False, linestyle="none") for i, n in enumerate(odors)]
        )

        self._save_or_show(normalization_names, save_fig_root, save_fig_prefix)

    def plot_final_odor_data_waterfall_by_odor(
            self, odor_name: str, normalization_names: set, ax: plt.Axes = None,
            save_fig_root: str | None = None, save_fig_prefix: str = ""
    ):
        add_labels = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        names, trials = self._get_final_odor_data(odor_name)

        trials = sorted(trials, key=lambda x: x[0])
        data = np.asarray([t[1] for t in trials])
        image = ax.imshow(data, origin="lower", interpolation="none", aspect="auto")
        ax.set_xticks([v[0] for v in names], [v[1] for v in names])
        if add_labels:
            ax.set_ylabel("Trials")
            ax.set_xlabel("Sensor")
            ax.set_title(f"Sensor intensity per {odor_name} trial")

            self._save_or_show(normalization_names, save_fig_root, save_fig_prefix + f"_{odor_name}")

        return image

    def plot_all_final_odor_data_waterfall_by_odor(
            self, normalization_names: set, n_rows=3,
            save_fig_root: str | None = None, save_fig_prefix: str = ""
    ):
        fig, axes, names = self._get_axes_for_odors(n_rows, sharex=True, sharey=False)
        for ax, name in zip(axes, names):
            image = self.plot_final_odor_data_waterfall_by_odor(name, normalization_names, ax=ax)
            ax.set_title(name)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(image, cax=cax, orientation='vertical')

        fig.supylabel("Trials")
        fig.supxlabel("Sensor")
        fig.suptitle("Sensor intensity per trial")
        self._save_or_show(normalization_names, save_fig_root, save_fig_prefix)

    def plot_final_odor_data_by_odor(
            self, odor_name: str, normalization_names: set, ax: plt.Axes = None,
            save_fig_root: str | None = None, save_fig_prefix: str = ""
    ):
        add_labels = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        names, trials = self._get_final_odor_data(odor_name)
        for trial, data in sorted(trials, key=lambda x: x[0]):
            ax.plot(np.arange(len(data)), data, marker="v", linestyle="dotted", label=f"T{trial}")

        ax.set_xticks([v[0] for v in names], [v[1] for v in names])
        if add_labels:
            ax.set_xlabel("Sensor")
            ax.set_ylabel("Intensity")
            ax.set_title(f"{odor_name} (lines are trials)")

            self._save_or_show(normalization_names, save_fig_root, save_fig_prefix + f"_{odor_name}")

    def plot_all_final_odor_data_by_odor(
            self, normalization_names: set, n_rows=3,
            save_fig_root: str | None = None, save_fig_prefix: str = ""
    ):
        fig, axes, names = self._get_axes_for_odors(n_rows, sharex=True, sharey=False)
        for ax, name in zip(axes, names):
            self.plot_final_odor_data_by_odor(name, normalization_names, ax=ax)
            ax.set_title(name)

        fig.supxlabel("Sensor")
        fig.supylabel("Intensity")
        fig.suptitle("Lines are trials")
        self._save_or_show(normalization_names, save_fig_root, save_fig_prefix)

    def plot_final_odor_data_by_sensor(
            self, sensor_id: str | None, normalization_names: set, average_trials: bool, ax: plt.Axes = None,
            save_fig_root: str | None = None, save_fig_prefix: str = ""
    ):
        add_labels = ax is None
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        sample_names, sample_arrays = self._get_final_odor_data_by_odor(sensor_id)
        labels = []
        for col in range(sample_arrays[0].shape[1]):
            ax.set_prop_cycle(None)
            offset = 0
            labels = []
            for name, data in zip(sample_names, sample_arrays):
                col_data = data[:, col]
                if average_trials:
                    col_data = np.median(col_data, keepdims=True)

                x = np.arange(len(col_data)) + offset
                line, = ax.plot(
                    x, col_data, marker="v", linestyle="dotted", label=name
                )
                labels.append(line)
                offset += len(col_data)

        if add_labels:
            if average_trials:
                ax.set_xlabel("Odors")
            else:
                ax.set_xlabel("Odors (with repeated trials)")
            ax.set_ylabel("Intensity")
            reps = "" if "dedup" in normalization_names else " (with replicates)"
            ax.set_title(f"Sensor{reps} intensity vs odors")
            fig.legend(handles=labels)
            self._save_or_show(normalization_names, save_fig_root, save_fig_prefix + f"_{sensor_id}")

        return labels

    def plot_all_final_odor_data_by_sensor(
            self, normalization_names: set, average_trials: bool, n_rows=3,
            save_fig_root: str | None = None, save_fig_prefix: str = ""
    ):
        fig, ax_map = self._get_fig_by_sensor(
            len(set(self.sensor_id)) + 2, n_rows, sharey=False, sharex=True, remaining_id="None"
        )
        remaining = ax_map.pop("None")

        labels = []
        for id_, ax in ax_map.items():
            labels = self.plot_final_odor_data_by_sensor(id_, normalization_names, average_trials, ax=ax)
            ax.set_title(f"Sensor {id_}")

        if average_trials:
            fig.supxlabel("Odors")
        else:
            fig.supxlabel("Odors (with repeated trials)")
        fig.supylabel("Intensity")
        reps = "" if "dedup" in normalization_names else " (with replicates)"
        fig.suptitle(f"Sensor{reps} intensity vs odors")
        fig.legend(handles=labels)
        self._save_or_show(normalization_names, save_fig_root, save_fig_prefix)

    def plot_final_odor_distance_by_sensor(
            self, sensor_id: str | None, plot_type: str, normalization_names: set, ax: plt.Axes = None, fig: plt.Figure = None,
            save_fig_root: str | None = None, save_fig_prefix: str = ""
    ):
        add_labels = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        sample_names, sample_arrays = self._get_final_odor_data_by_odor(sensor_id)
        sample_means = [np.mean(d, axis=0, keepdims=True) for d in sample_arrays]

        if plot_type == "dist_of_avg":
            samples = np.vstack(sample_means)
            dist = np.sqrt(np.sum(np.square(samples[:, np.newaxis, :] - samples[np.newaxis, :, :]), axis=2))
        elif plot_type == "avg_of_dist":
            dist = np.empty((len(sample_names), len(sample_names)))
            for i, avg in enumerate(sample_means):
                for j, values in enumerate(sample_arrays):
                    d = np.sqrt(np.sum(np.square(avg[:, np.newaxis, :] - values[np.newaxis, :, :]), axis=2))
                    dist[i, j] = np.median(d)
        else:
            raise ValueError(f"Unrecognized type {plot_type}")

        image = ax.imshow(dist, origin="lower", interpolation="none", aspect="auto")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(image, cax=cax, orientation='vertical')

        if add_labels:
            ax.set_xlabel("Odors")
            ax.set_ylabel("Odors")
            ax.set_title("Sensor response distance between odors")
            ax.legend(
                handles=[
                    mpatches.Patch(label=f"{i} - {n}", fill=False, linestyle="none")
                    for i, n in enumerate(sample_names)
                ]
            )
            self._save_or_show(normalization_names, save_fig_root, save_fig_prefix + f"_{sensor_id}")

        return sample_names

    def plot_all_final_odor_distance_by_sensor(
            self, normalization_names: set, plot_type: str, n_rows=3, save_fig_root: str | None = None,
            save_fig_prefix: str = ""
    ):
        fig, ax_map = self._get_fig_by_sensor(
            len(set(self.sensor_id)) + 1, n_rows, sharey=False, sharex=True, remaining_id="None"
        )
        remaining = ax_map.pop("None")

        for id_, ax in ax_map.items():
            self.plot_final_odor_distance_by_sensor(id_, plot_type, normalization_names, ax=ax, fig=fig)
            ax.set_title(f"Sensor {id_}")

        sample_names = self.plot_final_odor_distance_by_sensor(
            None, plot_type, normalization_names, ax=remaining[-1], fig=fig
        )
        remaining[-1].set_title(f"All sensors")

        fig.supxlabel("Odors")
        fig.supylabel("Odors")
        fig.suptitle("Sensor response distance between odors")
        fig.legend(
            handles=[
                mpatches.Patch(label=f"{i} - {n}", fill=False, linestyle="none") for i, n in enumerate(sample_names)
            ]
        )
        self._save_or_show(normalization_names, save_fig_root, save_fig_prefix)

    def plot_final_odor_data_class_probability(
            self, normalization_names: set, split: float, excluded_records: set = None,
            save_fig_root: str | None = None, save_fig_prefix: str = ""
    ):
        excluded_records = excluded_records or set()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        odors = list(sorted(set(r.sample_name for r in self.records if r.sample_name not in excluded_records)))
        odor_codes = {name: i for i, name in enumerate(odors)}

        all_data = []
        data_class = []
        train_data = []
        train_class = []
        test_data = []
        trial_count = 0
        trial_ticks = []
        test_count = 0
        test_ticks = []
        for class_, odor in enumerate(odors):
            sensor_names, trials = self._get_final_odor_data(odor)
            data = np.asarray([t[1] for t in trials])

            all_data.append(data)
            data_class.extend([class_,] * len(data))

            trial_ticks.append((trial_count, odor_codes[odor]))
            trial_count += len(data)

            n = int(round(len(data) * split))
            rem = len(data) - n
            train_data.append(data[:n, :])
            test_data.append(data[n:, :])
            train_class.extend([class_,] * n)

            if rem:
                test_ticks.append((test_count, odor_codes[odor]))
                test_count += rem

        all_data = np.vstack(all_data)
        data_class = np.asarray(data_class)

        clf = GaussianNB()
        clf.fit(all_data, data_class)
        probs = clf.predict_proba(all_data)

        image = ax1.imshow(probs.T, origin="lower", interpolation="none", aspect="auto")

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(image, cax=cax, orientation='vertical')

        ax1.set_xticks([v[0] for v in trial_ticks], [v[1] for v in trial_ticks])
        ax1.set_yticks(list(range(len(odors))), list(range(len(odors))))
        ax1.set_ylabel("Odor classes")
        ax1.set_xlabel("Odor trials")
        ax1.set_title("Odor class trial train probability")

        clf = GaussianNB()
        clf.fit(np.vstack(train_data), np.asarray(train_class))
        probs = clf.predict_proba(np.vstack(test_data))

        image = ax2.imshow(probs.T, origin="lower", interpolation="none", aspect="auto")

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(image, cax=cax, orientation='vertical')

        ax2.set_xticks([v[0] for v in test_ticks], [v[1] for v in test_ticks])
        ax2.set_yticks(list(range(len(odors))), list(range(len(odors))))
        ax2.set_ylabel("Odor classes")
        ax2.set_xlabel("Odor trials")
        ax2.set_title("Odor class trial test probability")

        fig.legend(
            handles=[mpatches.Patch(label=f"{i} - {n}", fill=False, linestyle="none") for i, n in enumerate(odors)]
        )

        self._save_or_show(normalization_names, save_fig_root, save_fig_prefix)

    def plot_all_final_odor_data_across_time(
            self, n_sensors, normalization_names: set, n_rows=3, save_fig_root: str | None = None, save_fig_prefix: str = ""):
        records = sorted(self.records, key=lambda r: r.baseline.unix_time[0])
        fig, ax_map = self._get_fig_by_sensor(n_sensors, n_rows, sharey=False, sharex=True)
        cmap = plt.get_cmap('tab20')
        color_names = {name: cmap(i % cmap.N) for i, name in enumerate(self.get_sample_names(True))}

        for i, record in enumerate(records):
            data = record.get_final_odor_sensor_data(normalization=set())
            for sensor, values in sorted(data.items(), key=lambda x: self.sensor_id.index(x[0])):
                ax_map[sensor].plot([i], np.median(np.abs(values)), '*', color=color_names[record.sample_name])

        for sensor, ax in ax_map.items():
            ax.set_title(f"Sensor {sensor}")
        fig.supxlabel("Trial order")
        fig.supylabel("Final odor intensity")
        fig.suptitle("Sensor odor intensity for all trials")
        self._save_or_show(normalization_names, save_fig_root, save_fig_prefix)

    def plot_sensor_binding_across_time(self, n_sensors, n_rows=3, end_duration=5):
        records = sorted(self.records, key=lambda r: r.baseline.unix_time[0])
        fig, ax_map = self._get_fig_by_sensor(n_sensors, n_rows, sharey=False, sharex=True)

        for i, record in enumerate(records):
            pre_slopes = record.estimate_baseline_slope()
            post_slopes = record.estimate_iti_slope(end_duration)

            for sensor_id, ax in ax_map.items():
                ax.plot([i], pre_slopes[sensor_id], '*', color='g')
                ax.plot([i], post_slopes[sensor_id], '*', color='b')

        fig.legend(handles=[mpatches.Patch(color="g", label="Baseline"), mpatches.Patch(color="b", label="ITI")])
        for sensor, ax in ax_map.items():
            ax.set_title(f"Sensor {sensor}")
        fig.supxlabel("Trial order")
        fig.supylabel("Slope")
        fig.suptitle("Sensor slopes for all trials")
        plt.show()

    def plot_sensor_binding_by_cycle_odor(
            self, n_sensors, n_rows=3, end_duration=5, use_post_data=True, abs_threshold=None
    ):
        records = sorted(self.records, key=lambda r: r.baseline.unix_time[0])
        fig, ax_map = self._get_fig_by_sensor(n_sensors, n_rows, sharey=False, sharex=True)
        cmap = plt.get_cmap('tab20')

        last_run = None
        last_record = None
        run_count = 0
        odor_color = {}
        pre_data = defaultdict(partial(defaultdict, list))
        post_data = defaultdict(partial(defaultdict, list))

        for record in records:
            pre_slopes = record.estimate_baseline_slope()
            post_slopes = record.estimate_iti_slope(end_duration)

            if record.run_id != last_run:
                last_run = record.run_id
                run_count += 1
                last_record = record
                continue

            if use_post_data:
                name = f"{run_count}.{record.sample_name}"
            else:
                name = f"{run_count}.{last_record.sample_name}"
            last_record = record

            if name not in odor_color:
                odor_color[name] = cmap(len(odor_color) % cmap.N)

            for sensor_id, val in pre_slopes.items():
                pre_data[sensor_id][name].append(val)
                post_data[sensor_id][name].append(post_slopes[sensor_id])
                # ax.plot([record.cycle], pre_slopes[sensor_id], '.', color=color)
                # ax.plot([record.cycle], post_slopes[sensor_id], '*', color=color)

        data = post_data if use_post_data else pre_data
        for sensor_id, odors in data.items():
            for name, values in odors.items():
                if abs_threshold is None or np.max(np.abs(values)) >= abs_threshold:
                    ax_map[sensor_id].plot(values, '-', color=odor_color[name])

        fig.legend(handles=[mpatches.Patch(color=c, label=n) for n, c in odor_color.items()], loc="upper center", ncols=8)
        for sensor, ax in ax_map.items():
            ax.set_title(f"Sensor {sensor}")
        fig.supxlabel("Trial")
        fig.supylabel("Slope")
        fig.suptitle("Sensor slopes per odor")
        plt.show()

    def write_sensor_binding_by_cycle_odor(self, filename, end_duration=5, sensor_avg_f=None):
        records = sorted(self.records, key=lambda r: r.baseline.unix_time[0])

        last_run = None
        last_record = None
        run_count = 0

        sensor_ids = sorted(set(self.sensor_id))
        header = ["Count", "Run", "Last odor", "Odor", ]
        if sensor_avg_f is None:
            header.extend(map(str, sensor_ids))
            header.extend(map(str, sensor_ids))
        else:
            header.extend(["Pre", "Post"])

        with open(filename, "w", newline="") as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(header)

            for i, record in enumerate(records):
                pre_slopes = record.estimate_baseline_slope()
                post_slopes = record.estimate_iti_slope(end_duration)

                if record.run_id != last_run:
                    last_run = record.run_id
                    run_count += 1
                    last_record = None

                last_name = "" if last_record is None else last_record.sample_name
                row = [i, run_count, last_name, record.sample_name]
                if sensor_avg_f is None:
                    for sensor_id in sensor_ids:
                        row.append(abs(pre_slopes[sensor_id]))
                    for sensor_id in sensor_ids:
                        row.append(abs(post_slopes[sensor_id]))
                else:
                    row.append(abs(sensor_avg_f([pre_slopes[sensor_id] for sensor_id in sensor_ids]).item()))
                    row.append(abs(sensor_avg_f([post_slopes[sensor_id] for sensor_id in sensor_ids]).item()))

                writer.writerow(map(str, row))
                last_record = record

    def _gather_records_data(self, excluded_records: set[str] = None) -> np.ndarray:
        excluded_records = excluded_records or set()
        aggregate_data = []
        for record in self.records:
            if record.sample_name in excluded_records:
                continue
            row, sensor_ids = record.get_flat_final_odor_sensor_data(normalization=set())
            aggregate_data.append(row)

        data = np.vstack(aggregate_data)
        return data

    def compute_sensor_pca(self, n_dim=2, excluded_records: set[str] = None):
        data = self._gather_records_data(excluded_records)

        pca = PCA(n_components=n_dim)
        pca.fit(data)

        return pca

    def process_records_pca(self, records: list[AryballeRecord], pca: PCA) -> None:
        labels = list(map(str, range(pca.n_components_)))
        self.records = [r.record_from_transform(pca.transform, labels) for r in records]

    def process_sensors_greatest_var(self, n_dim=2, excluded_records: set[str] = None):
        data = self._gather_records_data(excluded_records)

        var = np.std(data, axis=0, ddof=1) / np.mean(data, axis=0)
        order = np.argsort(var)[::-1]
        if n_dim is not None:
            order = order[:n_dim]

        sensor_id = [self.sensor_id[i] for i in order]
        sensor_id_unique = [self.sensor_id_unique[i] for i in order]

        def extract(arr):
            return arr[:, order]

        self.records = [r.record_from_transform(extract, sensor_id, sensor_id_unique) for r in self.records]

    def process_sensors_norm_var(self, excluded_records: set[str] = None):
        data = self._gather_records_data(excluded_records)
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, ddof=1, keepdims=True)

        def process(arr):
            valid = np.logical_not(np.isclose(std, 0))
            return np.divide(arr - mean, std, out=np.zeros_like(arr), where=valid)

        self.records = [r.record_from_transform(process, self.sensor_id, self.sensor_id_unique) for r in self.records]

    def plot_sensor_pca(
            self, pca, normalization_names: set, n_dim=2, added_records=None, save_fig_root: str | None = None,
            save_fig_prefix: str = ""):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d" if n_dim == 3 else None)
        cmap = plt.get_cmap("tab10")
        odor_color = {}

        for markers, records in ((("*", "."), self.records), (("h", "o"), added_records or [])):
            for record in records:
                row, sensor_ids = record.get_flat_final_odor_sensor_data(normalization=set())
                proj = pca.transform(row[np.newaxis, :])

                name = record.sample_name
                pure = "pure" in name
                pa10 = "10pa" in name or "10Pa" in name
                if pure or pa10:
                    name = name[:-5]
                if name not in odor_color:
                    odor_color[name] = cmap(len(odor_color) % cmap.N)
                color = odor_color[name]
                # only pa10 may be marked, pure could be absent
                marker = markers[0] if not pa10 else markers[1]

                ax.scatter(*proj[0, :], marker=marker, color=color)

        ax.legend(handles=[mpatches.Patch(color=c, label=t) for t, c in odor_color.items()])
        ax.set_xlabel("PCA 1")
        ax.set_xlabel("PCA 2")
        fig.suptitle(f"PCA - explained variance = {sum(pca.explained_variance_ratio_):0.2f}")
        self._save_or_show(normalization_names, save_fig_root, save_fig_prefix)

    def get_sample_names(self, sort=True) -> list[str]:
        if not sort:
            return list(set(r.sample_name for r in self.records))

        sorted_odors = []
        for record in sorted(self.records, key=lambda r: r.baseline.unix_time[0]):
            if record.sample_name not in sorted_odors:
                sorted_odors.append(record.sample_name)
        return sorted_odors

    def compute_h2o_affinity(self, h2o_name="Water", remove_h2o_records=False):
        water = []
        for record in self.records:
            if record.sample_name == h2o_name:
                water.append(record)
        if not water:
            raise ValueError("Unable to find water records")

        h2o_affinity = np.vstack([record.compute_water_h2o_affinity() for record in water])
        h2o_affinity = np.median(h2o_affinity, axis=0)

        for record in self.records:
            record.h2o_affinity = h2o_affinity
        self.h2o_affinity = h2o_affinity

        if remove_h2o_records:
            for record in self.records[:]:
                if record.sample_name == h2o_name:
                    self.records.remove(record)

    def remove_first_trial_records(self):
        for record in self.records[:]:
            if record.cycle == 1:
                self.records.remove(record)

    def compute_records_desorption_offset_line(self):
        record = self.records[0]
        for next_record in self.records[1:]:
            record.compute_next_record_slope_offset(next_record)
            record = next_record

    def compute_final_odor_distance(self, filename: str):
        odors = list(sorted(set(r.sample_name for r in self.records)))

        names = []
        data = []
        for odor in odors:
            for record in self.records:
                if record.sample_name != odor:
                    continue

                items = record.get_final_odor_sensor_data(normalization=set())
                all_values = []
                for sensor, values in sorted(items.items(), key=lambda x: x[0]):
                    all_values.append(values)

                data.append(np.concatenate(all_values))
                names.append(f"{record.sample_name} T{record.cycle}")

        data = np.asarray(data)
        table = np.empty((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                table[i, j] = np.linalg.norm(data[i, :] - data[j, :], 2)

        table -= table.min()
        table /= table.max()

        header = ",".join([n.replace(",", "_") for n in names])
        np.savetxt(filename, table, delimiter=",", header=header)


def norm_combinations(*norms: str, dedup=False) -> list[set[str]]:
    all_combinations = []
    for i in range(len(norms) + 1):
        for opts in combinations(norms, i):
            opts = set(opts)
            if "norm_to_area" in opts:
                opts.add("norm_range")

            if dedup:
                opts.add("dedup")

            all_combinations.append(opts)
    return all_combinations


if __name__ == "__main__":
    records = AryballeRecords()

    n_rows = 3
    prefix = "aryballe"
    fig_root = r"C:\Users\Matthew Einhorn\Downloads\figs"

    records.add_aryballe_csv_records(r"C:\Users\Matthew Einhorn\Downloads\sensors\CTTYN_40s_10min.sensograms.csv")
    # records.add_prism_xlsx_records(
    #     r"C:\Users\Matthew Einhorn\Downloads\Prism 5\data\20230411.xlsx",
    #     r"C:\Users\Matthew Einhorn\Downloads\Prism 5\data\20230503.xlsx",
    #     r"C:\Users\Matthew Einhorn\Downloads\Prism 5\data\20230426.xlsx",
    #     r"C:\Users\Matthew Einhorn\Downloads\Prism 5\data\20230403.xlsx",
    #     r"C:\Users\Matthew Einhorn\Downloads\Prism 5\data\20230419.xlsx",
    #     r"C:\Users\Matthew Einhorn\Downloads\Prism 5\data\20230406.xlsx",
    # )
    raw_records = records.records
    # for record in records.records:
    #     print(record)

    # records.compute_records_desorption_offset_line()
    records.remove_first_trial_records()
    records.compute_h2o_affinity(remove_h2o_records=True, h2o_name="Water")

    for dedup in (True, False):
        all_combinations = norm_combinations("blank", "norm_to_area", "humidity", dedup=dedup)
        if dedup:
            root_ = fig_root + r"\dedup"
        else:
            root_ = fig_root + r"\with_dup"

        for proc in ("normal", "pca", "var_ordered"):
            root = root_ + fr"\{proc}"

            for norms in all_combinations:
                print(dedup, proc, norms)
                records.process_records(raw_records, norms)

                if proc == "var_ordered":
                    records.process_sensors_greatest_var(n_dim=None, excluded_records={"Blank"})
                elif proc == "pca":
                    pca = records.compute_sensor_pca(n_dim=8, excluded_records={"Blank"})
                    records.process_records_pca(records.records, pca)

                records.plot_all_final_odor_data_by_odor(
                    norms, save_fig_root=root + r"\by_odor", save_fig_prefix=prefix, n_rows=n_rows
                )
                records.plot_all_final_odor_data_waterfall_by_odor(
                    norms, save_fig_root=root + r"\by_odor_waterfall", save_fig_prefix=prefix, n_rows=n_rows
                )
                records.plot_final_odor_data_waterfall_all_odors(
                    norms, False, save_fig_root=root + r"\all_odors_and_sensors", save_fig_prefix=prefix
                )
                records.plot_final_odor_data_waterfall_all_odors(
                    norms, True, save_fig_root=root + r"\all_odors_and_sensors_avg_trial",
                    save_fig_prefix=prefix
                )
                records.plot_all_final_odor_distance_by_sensor(
                    norms, "dist_of_avg", save_fig_root=root + r"\by_odor_dist_of_avg", save_fig_prefix=prefix,
                    n_rows=n_rows
                )
                records.plot_all_final_odor_distance_by_sensor(
                    norms, "avg_of_dist", save_fig_root=root + r"\by_odor_avg_of_dist", save_fig_prefix=prefix,
                    n_rows=n_rows
                )
                records.plot_all_final_odor_data_by_sensor(
                    norms, average_trials=False, save_fig_root=root + r"\by_sensor", save_fig_prefix=prefix,
                    n_rows=n_rows
                )
                records.plot_all_final_odor_data_by_sensor(
                    norms, average_trials=True, save_fig_root=root + r"\by_sensor_avg_trial", save_fig_prefix=prefix,
                    n_rows=n_rows
                )
                records.plot_final_odor_data_class_probability(
                    norms, 0.8, excluded_records={"Blank"},
                    save_fig_root=root + r"\naive_bayes_classification", save_fig_prefix=prefix
                )
