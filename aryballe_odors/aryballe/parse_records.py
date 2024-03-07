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


class TrialSegment:

    sensor_id: list[int]

    sensor_pos: list[tuple[str, int]]

    time_running: np.ndarray

    time: np.ndarray

    unix_time: np.ndarray

    humidity: np.ndarray

    temp: np.ndarray

    data: np.ndarray

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

    record: int

    run_id: str

    run_name: str

    device_id: str

    sample_name: str

    cycle: int

    sensor_id: list[int]

    sensor_pos: list[tuple[str, int]]

    baseline: TrialSegment

    odor: TrialSegment

    iti: TrialSegment

    tags: set[str]

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

    def _get_data(self, section, reset_time=True, subtract_baseline=False):
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
        if subtract_baseline:
            data -= self._get_baseline_offset()

        return data, time

    def _get_sensor_masks(self) -> dict[int, np.ndarray]:
        sensor_ids = self.sensor_id

        masks = {}
        for sensor_id in set(sensor_ids):
            masks[sensor_id] = np.array([val == sensor_id for val in sensor_ids], dtype=np.bool_)

        return masks

    def _get_baseline_offset(self):
        return np.median(self.baseline.data, axis=0, keepdims=True)

    def _normalize_data(self, data):
        offset = self._get_baseline_offset()
        scale = np.median(np.abs(self.odor.data[-1, :] - offset[0, :]))

        if len(data.shape) == 2:
            return (data - offset) / scale
        return (data - offset[0, :]) / scale

    def get_data_by_sensor(
            self, section=None, reset_time=True, subtract_baseline=False, normalize=False
    ) -> tuple[dict[int, np.ndarray], np.ndarray]:
        data, time = self._get_data(
            section, reset_time=reset_time, subtract_baseline=subtract_baseline and not normalize)

        if normalize:
            data = self._normalize_data(data)

        sensor_data = {}
        for sensor_id, mask in self._get_sensor_masks().items():
            sensor_data[sensor_id] = data[:, mask]

        return sensor_data, time

    def get_final_odor_sensor_data(self, subtract_baseline=False, normalize=False) -> dict[int, np.ndarray]:
        if normalize:
            data = self._normalize_data(self.odor.data[-1, :])
        elif subtract_baseline:
            data = self.odor.data[-1, :] - self._get_baseline_offset()[0, :]
        else:
            data = self.odor.data[-1, :]

        sensor_data = {}
        for sensor_id, mask in self._get_sensor_masks().items():
            sensor_data[sensor_id] = data[mask]

        return sensor_data

    def get_flat_final_odor_sensor_data(self, subtract_baseline=False, normalize=False) -> np.ndarray:
        if normalize:
            data = self._normalize_data(self.odor.data[-1, :])
        elif subtract_baseline:
            data = self.odor.data[-1, :] - self._get_baseline_offset()[0, :]
        else:
            data = self.odor.data[-1, :]

        return data

    def estimate_baseline_slope(self) -> dict[int, np.ndarray]:
        data = self.baseline.data

        slopes = {}
        for sensor_id, mask in self._get_sensor_masks().items():
            res = stats.linregress(self.baseline.time, np.median(data[:, mask], axis=1))
            slopes[sensor_id] = res.slope

        return slopes

    def estimate_iti_slope(self, end_duration=None) -> dict[int, np.ndarray]:
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

    records: list[AryballeRecord]

    @property
    def sensor_id(self) -> list[int]:
        return self.records[0].sensor_id

    @property
    def sensor_pos(self) -> list[tuple[str, int]]:
        return self.records[0].sensor_pos

    def __init__(self):
        self.records = []

    def add_records(self, filename: str):
        self.records.extend(self._parse_records_csv(filename))

    def _split_rows_by_id(self, data: list, idx: int) -> list[tuple[int, int]]:
        last = None
        indices = []

        for i, row in enumerate(data):
            if row[idx] != last:
                indices.append(i)
                last = row[idx]

        return list(zip(indices, indices[1:] + [len(data)]))

    def _parse_records_csv(self, filename: str) -> list[AryballeRecord]:
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

    def plot_by_sensor_id(
            self, n_sensors, odor_name: str, section=None, n_rows=3, subtract_baseline=False, normalize=False):
        fig, ax_map = self._get_fig_by_sensor(n_sensors, n_rows, sharey=True, sharex=True)
        ax_count = defaultdict(int)

        for record in self.records:
            if record.sample_name != odor_name:
                continue

            data, time = record.get_data_by_sensor(section, subtract_baseline=subtract_baseline, normalize=normalize)
            for sensor, values in data.items():
                ax_map[sensor].plot(time, values, '-')
                ax_count[sensor] += values.shape[1]

        for sensor, ax in ax_map.items():
            ax.set_title(f"Sensor {sensor} ({ax_count[sensor]} reps)")
        fig.supxlabel("Time (s)")
        fig.supylabel("Amplitude")
        fig.suptitle(odor_name)
        plt.show()

    def plot_final_odor_data_by_sensor(
            self, odor_name: str, subtract_baseline=False, ax: plt.Axes = None, normalize=False):
        add_labels = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        cmap = plt.get_cmap('tab10')
        color_names = {}
        axis_names = {}

        for record in self.records:
            if record.sample_name != odor_name:
                continue

            data = record.get_final_odor_sensor_data(subtract_baseline=subtract_baseline, normalize=normalize)
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
            plt.show()

    def plot_all_final_odor_data_by_sensor(self, subtract_baseline=False, normalize=False, n_rows=3):
        odors = list(sorted(set(r.sample_name for r in self.records)))

        n_cols = int(math.ceil(len(odors) / n_rows))
        fig, axs = plt.subplots(n_rows, n_cols, sharey=False, sharex=True)
        ax_flat = axs.flatten().tolist()

        for ax, name in zip(ax_flat, odors):
            self.plot_final_odor_data_by_sensor(name, subtract_baseline=subtract_baseline, ax=ax, normalize=normalize)
            ax.set_title(name)

        fig.supxlabel("Sensor")
        fig.supylabel("Amplitude")
        plt.show()

    def plot_all_final_odor_data_across_time(self, n_sensors, n_rows=3, subtract_baseline=False, normalize=False):
        records = sorted(self.records, key=lambda r: r.baseline.unix_time[0])
        fig, ax_map = self._get_fig_by_sensor(n_sensors, n_rows, sharey=False, sharex=True)
        cmap = plt.get_cmap('tab20')
        color_names = {name: cmap(i % cmap.N) for i, name in enumerate(self.get_sample_names(True))}

        for i, record in enumerate(records):
            data = record.get_final_odor_sensor_data(subtract_baseline=subtract_baseline, normalize=normalize)
            for sensor, values in sorted(data.items(), key=lambda x: x[0]):
                ax_map[sensor].plot([i], np.median(np.abs(values)), '*', color=color_names[record.sample_name])

        for sensor, ax in ax_map.items():
            ax.set_title(f"Sensor {sensor}")
        fig.supxlabel("Trial order")
        fig.supylabel("Final odor intensity")
        fig.suptitle("Sensor odor intensity for all trials")
        plt.show()

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

    def compute_sensor_pca(self, n_dim=2, subtract_baseline=False, normalize=False):
        aggregate_data = []
        for record in self.records:
            row = record.get_flat_final_odor_sensor_data(subtract_baseline=subtract_baseline, normalize=normalize)
            aggregate_data.append(row)
        data = np.vstack(aggregate_data)

        pca = PCA(n_components=n_dim)
        pca.fit(data)

        return pca

    def plot_sensor_pca(self, pca, n_dim=2, added_records=None, subtract_baseline=False, normalize=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d" if n_dim == 3 else None)
        cmap = plt.get_cmap("tab10")
        odor_color = {}

        for markers, records in ((("*", "."), self.records), (("h", "o"), added_records or [])):
            for record in records:
                row = record.get_flat_final_odor_sensor_data(subtract_baseline=subtract_baseline, normalize=normalize)
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
        print(pca.explained_variance_ratio_)
        ax.set_xlabel("PCA 1")
        ax.set_xlabel("PCA 2")
        fig.suptitle(f"PCA - explained variance = {sum(pca.explained_variance_ratio_):0.2f}")
        plt.show()

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
    records.add_records(r"C:\Users\Matthew Einhorn\Downloads\sensors\pure_vs_10pa.sensograms.csv")
    # records.add_records(r"C:\Users\Matthew Einhorn\Downloads\sensors\pure_long.sensograms.csv")
    # for record in records.records:
    #     print(record)
    # for name in records.get_sample_names(True):
    #     records.plot_by_sensor_id(9, name, section=None, normalize=False)
    #     records.plot_final_odor_data_by_sensor(name, normalize=False)
    # records.plot_by_sensor_id(9, "acetone 10Pa", section=None)
    # records.plot_final_odor_data_by_sensor("acetone 10Pa")
    records.plot_all_final_odor_data_by_sensor()
    # records.plot_all_final_odor_data_across_time(9, normalize=False)
    # records.plot_sensor_binding_across_time(9)
    # records.plot_sensor_binding_by_cycle_odor(9, use_post_data=True, abs_threshold=.06)
    # records.write_sensor_binding_by_cycle_odor(r"C:\Users\Matthew Einhorn\Downloads\sensors\sensor_binding_long_iti.csv")
    # pca = records.compute_sensor_pca(normalize=False)
    #
    # records2 = AryballeRecords()
    # records2.add_records(r"C:\Users\Matthew Einhorn\Downloads\sensors\pure_long.sensograms.csv")
    # records.plot_sensor_pca(pca, added_records=records2.records, normalize=False)
