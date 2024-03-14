import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.decomposition import PCA
import math
from functools import partial
from collections import defaultdict


HEADER_PAT = re.compile(r"([0-9]+)\[([A-Z]+)([0-9]+)]")

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

BLANK_SENSOR = 1


class TrialSegment:
    """Data of component of a trial.

    A trial is composed of e.g. a baseline measurement, analyte and, post analyte measurement. This is the data
    of a single segment, e.g. the baseline part of the trial.
    """

    sensor_id: list[int]
    """A 1d list of the same number of columns of `data`, each item is the id of the peptide. A peptide can be
    represented multiple times due to replicates.
    """

    sensor_pos: list[tuple[str, int]]
    """A 1d list of the same number of columns of `data`, each item is the position info of the peptide and unique.

    Position is represented by a letter and number, row column format. E.g. ("A", 2) meaning A2.
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

    def __init__(
            self, sensor_id, sensor_pos, relative_time, unix_time, humidity, temp, data):
        self.sensor_id = sensor_id
        self.sensor_pos = sensor_pos
        self.time_running = relative_time
        self.time = relative_time - relative_time[0]
        self.duration = self.time[-1].item()
        self.unix_time = unix_time
        self.humidity = humidity
        self.temp = temp
        self.data = data


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

    sensor_id: list[int]
    """Same as TrialSegment.sensor_id.
    """

    sensor_pos: list[tuple[str, int]]
    """Same as TrialSegment.sensor_pos.
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

    def __init__(
            self, record, run_id, run_name, device_id, sample_name,
            cycle, sensor_id, sensor_pos, baseline, odor, iti, tags):
        self.record = record
        self.run_id = run_id
        self.run_name = run_name
        self.device_id = device_id
        self.sample_name = sample_name
        self.cycle = cycle
        self.sensor_id = sensor_id
        self.sensor_pos = sensor_pos
        self.baseline = baseline
        self.odor = odor
        self.iti = iti
        self.tags = tags

    def __str__(self):
        return (
            f"<{id(self)}@AryballeRecord sample=\"{self.sample_name}\", cycle={self.cycle}, "
            f"times=({self.baseline.duration:0.2f},{self.odor.duration:0.2f},{self.iti.duration:0.2f}), "
            f"tags=[{','.join(sorted(self.tags))}]>"
        )

    def get_data(self, section, normalization: set, norm_odor_peak: bool, reset_time=True) -> tuple[np.ndarray, np.ndarray]:
        """Gets the 2d data (same axis as `TrialSegment.data`) and associated timestamps of the record.

        :param section: The data components - it concatenates the requested data. Can be None, which returns the same as
            `"baseline+odor+iti"`. Also accepts, `"baseline+odor"`, `"odor+iti"`, `"baseline"`, `"odor"`, and `"iti"`
            that returns a single array of the data.
        :param normalization: See `normalize_data`.
        :param reset_time: Whether to reset the timestamp of the first data row to zero.
        :return: A tuple of data and time. Data is time X sensors (rows, columns). Time is the 1d array of the
        associated continuous timestamps of the data.
        """
        match section:
            case None | "baseline+odor+iti":
                # all sections
                items = [self.baseline, self.odor, self.iti]
            case "baseline+odor":
                items = [self.baseline, self.odor]
            case "odor+iti":
                items = [self.odor, self.iti]
            case "baseline":
                items = [self.baseline]
            case "odor":
                items = [self.odor]
            case "iti":
                items = [self.iti]
            case _:
                raise ValueError(f"Unknown sections {section}")

        data = np.concatenate([getattr(item, "data") for item in items], axis=0)
        time = np.concatenate([getattr(item, "time_running") for item in items], axis=0)
        if reset_time:
            time -= time[0]

        data = self.normalize_data(data, normalization=normalization, norm_odor_peak=norm_odor_peak)

        return data, time

    def _get_sensor_masks(self) -> dict[int, np.ndarray]:
        """Gets a dict mapping sensor IDs to a mask of the columns of the repeats of this sensor (peptide).

        Each key is the ID of the sensor (e.g. 25). The corresponding value is the logical ndarray which has true values
        at the indices (columns) corresponding to the repeats of this peptide. Indexing `data` or `sensor_pos` will
        return the columns (values) of this sensor repeats.
        """
        sensor_ids = self.sensor_id

        masks = {}
        for sensor_id in set(sensor_ids):
            masks[sensor_id] = np.array([val == sensor_id for val in sensor_ids], dtype=np.bool_)

        return masks

    def get_baseline_offset(self, keep2dims=True) -> np.ndarray:
        """Returns the offset of each sensor to zero it.

        :param keep2dims: If True will keep the first dim to have a shape of 1 X N, otherwise just 1D N sized array.
        :return: An array, that is the offset for each sensor. Such that subtracting it from the corresponding sensor
            will keep the average value at zero, as computed from the baseline data.
        """
        return np.median(self.baseline.data, axis=0, keepdims=keep2dims)

    def normalize_data(self, data, normalization: set, norm_odor_peak: bool) -> np.ndarray:
        """Normalizes the input data.

        Normalization is computed by removing the offset computed from the baseline. Then, it's scaled by the magnitude
        at the end of the analyte period. It is computed as a single value for all the sensors.
        """
        d1 = len(data.shape) == 1
        blank_idx = np.array(self.sensor_id, dtype=np.int_) == BLANK_SENSOR
        if d1:
            data = data[np.newaxis, :]

        if "baseline" in normalization:
            offset = self.get_baseline_offset(keep2dims=True)
            data -= offset

        if "blank" in normalization:
            data -= np.median(data[:, blank_idx], axis=1, keepdims=True)

        if "norm" in normalization:
            norm_data = data
            if norm_odor_peak:
                norm_data = self.odor.data[-1, :][np.newaxis, :]

            medians = []
            for mask in self._get_sensor_masks().values():
                medians.append(np.median(norm_data[:, mask], axis=1, keepdims=True))
            scale = np.linalg.norm(np.hstack(medians), axis=1, keepdims=True)

            data = np.divide(data, scale, out=np.zeros_like(data), where=scale != 0)

        if d1:
            return data[0, :]
        return data

    def get_data_by_sensor(
            self, normalization: set, norm_odor_peak: bool, section=None, reset_time=True
    ) -> tuple[dict[int, np.ndarray], np.ndarray]:
        """Returns the sensor data, split by sensor (peptide) type.

        :param section: See `get_data`.
        :param normalization: See `normalize_data`.
        :param reset_time: See `get_data`.
        :return: Similar to `get_data`, but split into mapping for sensor (peptide) IDs to repeats of the peptide.
            I.e. keys are the IDs of the peptides. Each value is a 2d array, with rows being time, and columns are
            the repeats of only this peptide.
        """
        data, time = self.get_data(
            section, normalization=normalization, norm_odor_peak=norm_odor_peak, reset_time=reset_time)

        sensor_data = {}
        for sensor_id, mask in self._get_sensor_masks().items():
            sensor_data[sensor_id] = data[:, mask]

        return sensor_data, time

    def get_final_odor_sensor_data(self, normalization: set) -> dict[int, np.ndarray]:
        """Returns the sensor data from the end of the analyte (odor segment), split by sensor (peptide) type.

        :param normalization: See `normalize_data`.
        :return: The last data sample of the odor segment, split by peptide. I.e. a mapping of sensor (peptide) IDs to
            repeats of the peptide. Keys are the IDs of the peptides. Each value is a 1d array with the value of the
            repeats of only this peptide.
        """
        data = self.get_flat_final_odor_sensor_data(normalization=normalization)

        sensor_data = {}
        for sensor_id, mask in self._get_sensor_masks().items():
            sensor_data[sensor_id] = data[mask]

        return sensor_data

    def get_flat_final_odor_sensor_data(self, normalization: set) -> np.ndarray:
        """Returns the sensor data from the end of the analyte (odor segment).

        :param normalization: See `normalize_data`.
        :return: The last data sample of the odor segment as 1d array, where each value corresponds to a sensor
            in `sensor_id`.
        """
        data = self.normalize_data(self.odor.data[-1, :], normalization=normalization, norm_odor_peak=True)

        return data

    def estimate_baseline_slope(self) -> dict[int, np.ndarray]:
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

    def estimate_iti_slope(self, end_duration=None) -> dict[int, np.ndarray]:
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

    @property
    def sensor_id(self) -> list[int]:
        """See TrialSegment.sensor_id.
        """
        return self.records[0].sensor_id

    @property
    def sensor_pos(self) -> list[tuple[str, int]]:
        """See TrialSegment.sensor_pos.
        """
        return self.records[0].sensor_pos

    def __init__(self):
        self.records = []

    def add_csv_records(self, filename: str) -> None:
        """Adds the records from the given CSV file to `records`.

        :param filename: The CSV filename as exported from Aryballe hub.
        """
        self.records.extend(self.parse_records_csv(filename))

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

    def parse_records_csv(self, filename: str) -> list[AryballeRecord]:
        """Parses and returns the records from the given CSV file.

        :param filename: The CSV filename as exported from Aryballe hub.
        """
        with open(filename, 'r') as fh:
            reader = list(csv.reader(fh, delimiter=","))

        header = reader[0]

        n_sensors = 0
        sensor_id = []
        sensor_pos = []
        for item in header[SENSOR0:]:
            m = re.match(HEADER_PAT, item)
            if m is not None:
                n_sensors += 1
                id_, c, i = m.groups()
                sensor_id.append(int(id_))
                sensor_pos.append((c, int(i)))
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
            assert len(segments) == 3, "expected 3 segments in trial"

            trials = []
            for rec_s, rec_e in segments:
                segment_data = record_data[rec_s: rec_e]
                rel_time = np.array([row[RELATIVE_TIME] for row in segment_data], dtype=np.float32)
                unix_time = np.array([row[UNIX_TIME] for row in segment_data], dtype=np.float64)
                humidity = np.array([row[HUMIDITY_VAL] for row in segment_data], dtype=np.float32)
                temp = np.array([row[TEMP_VAL] for row in segment_data], dtype=np.float32)
                data = np.array([row[SENSOR0: SENSOR0 + n_sensors] for row in segment_data], dtype=np.float32)

                trial = TrialSegment(
                    sensor_id=sensor_id, sensor_pos=sensor_pos, relative_time=rel_time, unix_time=unix_time,
                    humidity=humidity, temp=temp, data=data
                )
                trials.append(trial)

            baseline, odor, iti = trials

            record_tags = set()
            for i, name in tags:
                if row0[i] == 'true':
                    record_tags.add(name)

            record = AryballeRecord(
                record=int(row0[REC_NUM]), run_id=int(row0[RUN_ID]), run_name=row0[RUN_NAME], device_id=row0[DEVICE_ID],
                sample_name=row0[ITEM_NAME], cycle=int(row0[CYCLE_NUM]), sensor_id=sensor_id, sensor_pos=sensor_pos,
                baseline=baseline, odor=odor, iti=iti, tags=record_tags
            )

            records.append(record)

        return records

    def _get_fig_by_sensor(self, n_sensors, n_rows, sharey, sharex) -> tuple[plt.Figure, dict[int, plt.Axes]]:
        n_cols = int(math.ceil(n_sensors / n_rows))
        fig, axs = plt.subplots(n_rows, n_cols, sharey=sharey, sharex=sharex)
        ax_flat = axs.flatten().tolist()
        ax_map = {i: ax_flat.pop(0) for i in sorted(set(self.sensor_id))}

        return fig, ax_map

    def _save_or_show(self, normalization: set, save_fig_root: str | None = None, save_fig_prefix: str = ""):
        if save_fig_root:
            norm = "none"
            if normalization:
                norm = ",".join(sorted(normalization))

            fig = plt.gcf()
            fig.set_size_inches(15, 9)
            fig.savefig(
                f"{save_fig_root}/{save_fig_prefix}_norm={norm}.png", bbox_inches='tight',
                dpi=300
            )
            plt.close()
        else:
            plt.show()

    def plot_by_sensor_id(
            self, n_sensors, odor_name: str, normalization: set, norm_odor_peak: bool, section=None, n_rows=3,
            save_fig_root: str | None = None, save_fig_prefix: str = ""
    ):
        fig, ax_map = self._get_fig_by_sensor(n_sensors, n_rows, sharey=True, sharex=True)
        ax_count = defaultdict(int)

        for record in self.records:
            if record.sample_name != odor_name:
                continue

            data, time = record.get_data_by_sensor(
                normalization=normalization, norm_odor_peak=norm_odor_peak, section=section, reset_time=True)
            for sensor, values in data.items():
                ax_map[sensor].plot(time, values, '-')
                ax_count[sensor] += values.shape[1]

        for sensor, ax in ax_map.items():
            ax.set_title(f"Sensor {sensor} ({ax_count[sensor]} reps)")
        fig.supxlabel("Time (s)")
        fig.supylabel("Amplitude")
        fig.suptitle(odor_name)

        self._save_or_show(normalization, save_fig_root, save_fig_prefix + f"_{odor_name}")

    def plot_final_odor_data_by_sensor(
            self, odor_name: str, normalization: set, ax: plt.Axes = None, save_fig_root: str | None = None,
            save_fig_prefix: str = ""
    ):
        add_labels = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        cmap = plt.get_cmap('tab10')
        color_names = {}
        axis_names = {}

        for record in self.records:
            if record.sample_name != odor_name:
                continue

            data = record.get_final_odor_sensor_data(normalization=normalization)
            start = 0
            for sensor, values in sorted(data.items(), key=lambda x: x[0]):
                n = len(values)
                color_names[f"Trial {record.cycle}"] = c = cmap(record.cycle % cmap.N)
                axis_names[sensor] = start + (n - 1) / 2

                ax.plot(range(start, start + n), values, '*', color=c)
                start += n + 2

        xtics = list(sorted(axis_names))
        ax.set_xticks([axis_names[k] for k in xtics], labels=list(map(str, xtics)))

        if add_labels:
            ax.legend(handles=[mpatches.Patch(color=c, label=t) for t, c in color_names.items()])
            ax.set_xlabel("Sensor")
            ax.set_ylabel("Amplitude")
            ax.set_title(odor_name)

            self._save_or_show(normalization, save_fig_root, save_fig_prefix + f"_{odor_name}")

    def plot_all_final_odor_data_by_sensor(
            self, normalization: set, n_rows=3, save_fig_root: str | None = None, save_fig_prefix: str = ""):
        odors = list(sorted(set(r.sample_name for r in self.records)))

        n_cols = int(math.ceil(len(odors) / n_rows))
        fig, axs = plt.subplots(n_rows, n_cols, sharey=False, sharex=True)
        ax_flat = axs.flatten().tolist()

        for ax, name in zip(ax_flat, odors):
            self.plot_final_odor_data_by_sensor(name, normalization, ax=ax)
            ax.set_title(name)

        fig.supxlabel("Sensor")
        fig.supylabel("Amplitude")
        self._save_or_show(normalization, save_fig_root, save_fig_prefix)

    def plot_all_final_odor_data_across_time(
            self, n_sensors, normalization: set, n_rows=3, save_fig_root: str | None = None, save_fig_prefix: str = ""):
        records = sorted(self.records, key=lambda r: r.baseline.unix_time[0])
        fig, ax_map = self._get_fig_by_sensor(n_sensors, n_rows, sharey=False, sharex=True)
        cmap = plt.get_cmap('tab20')
        color_names = {name: cmap(i % cmap.N) for i, name in enumerate(self.get_sample_names(True))}

        for i, record in enumerate(records):
            data = record.get_final_odor_sensor_data(normalization=normalization)
            for sensor, values in sorted(data.items(), key=lambda x: x[0]):
                ax_map[sensor].plot([i], np.median(np.abs(values)), '*', color=color_names[record.sample_name])

        for sensor, ax in ax_map.items():
            ax.set_title(f"Sensor {sensor}")
        fig.supxlabel("Trial order")
        fig.supylabel("Final odor intensity")
        fig.suptitle("Sensor odor intensity for all trials")
        self._save_or_show(normalization, save_fig_root, save_fig_prefix)

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

    def compute_sensor_pca(self, normalization: set, n_dim=2):
        aggregate_data = []
        for record in self.records:
            row = record.get_flat_final_odor_sensor_data(normalization=normalization)
            aggregate_data.append(row)
        data = np.vstack(aggregate_data)

        pca = PCA(n_components=n_dim)
        pca.fit(data)

        return pca

    def plot_sensor_pca(
            self, pca, normalization: set, n_dim=2, added_records=None, save_fig_root: str | None = None,
            save_fig_prefix: str = ""):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d" if n_dim == 3 else None)
        cmap = plt.get_cmap("tab10")
        odor_color = {}

        for markers, records in ((("*", "."), self.records), (("h", "o"), added_records or [])):
            for record in records:
                row = record.get_flat_final_odor_sensor_data(normalization=normalization)
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
        self._save_or_show(normalization, save_fig_root, save_fig_prefix)

    def get_sample_names(self, sort=True) -> list[str]:
        if not sort:
            return list(set(r.sample_name for r in self.records))

        sorted_odors = []
        for record in sorted(self.records, key=lambda r: r.baseline.unix_time[0]):
            if record.sample_name not in sorted_odors:
                sorted_odors.append(record.sample_name)
        return sorted_odors


if __name__ == "__main__":
    records = AryballeRecords()
    fig_root = r"C:\Users\Matthew Einhorn\Downloads\figs"
    records.add_csv_records(r"C:\Users\Matthew Einhorn\Downloads\sensors\pure_vs_10pa.sensograms.csv")
    # records.add_csv_records(r"C:\Users\Matthew Einhorn\Downloads\sensors\pure_long.sensograms.csv")
    # for record in records.records:
    #     print(record)
    for norms in (set(), {"baseline", }, {"baseline", "blank"}, {"baseline", "blank", "norm"}):
        for name in records.get_sample_names(sort=True):
            records.plot_by_sensor_id(
                9, name, normalization=norms, norm_odor_peak=False, section=None, save_fig_root=fig_root,
                save_fig_prefix="odor_full_trace"
            )
            if "norm" in norms:
                records.plot_by_sensor_id(
                    9, name, normalization=norms, norm_odor_peak=True, section=None, save_fig_root=fig_root,
                    save_fig_prefix="odor_full_trace_peak_normed"
                )
            records.plot_final_odor_data_by_sensor(
                name, normalization=norms, save_fig_root=fig_root, save_fig_prefix="odor_final_peak")
        # records.plot_by_sensor_id(9, "acetone pure", section=None)
        # records.plot_final_odor_data_by_sensor("acetone pure")
        records.plot_all_final_odor_data_by_sensor(
            normalization=norms, save_fig_root=fig_root, save_fig_prefix="by_sensor_by_odor")
        records.plot_all_final_odor_data_across_time(
            9, normalization=norms, save_fig_root=fig_root, save_fig_prefix="by_sensor_by_trial")
        # records.plot_sensor_binding_across_time(9)
        # records.plot_sensor_binding_by_cycle_odor(9, use_post_data=True, abs_threshold=.06)
        # records.write_sensor_binding_by_cycle_odor(r"C:\Users\Matthew Einhorn\Downloads\sensors\sensor_binding_long_iti.csv")

        pca = records.compute_sensor_pca(normalization=norms)
        records.plot_sensor_pca(
            pca, normalization=norms, save_fig_root=fig_root, save_fig_prefix="pca_1st_set_only")

        records2 = AryballeRecords()
        records2.add_csv_records(r"C:\Users\Matthew Einhorn\Downloads\sensors\pure_long.sensograms.csv")
        records.plot_sensor_pca(
            pca, normalization=norms, added_records=records2.records, save_fig_root=fig_root,
            save_fig_prefix="pca_1st_set_with_2nd_only_tested")
