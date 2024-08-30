import os
import yaml
import time
import datetime
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import swmmtoolbox.swmmtoolbox as swm
from datetime import datetime, timedelta

load_dotenv()
swmm_executable_path = Path(os.getenv("SWMM_EXECUTABLE_PATH"))


def import_config_from_yaml(config_file):
    """
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    """
    with open(config_file) as f:
        data = yaml.safe_load(f)

    return data


# Tabulator ----------------------------------------------------------------------------------------------
def tabulator(list_of_strings):
    new_list = []
    for i in list_of_strings:
        new_list.append(i)
        new_list.append("\t")
    new_list[-1] = "\n"
    return new_list


# From rain generator code -------------------------------------------------------------------------------
def get_multiple_alt_blocks_rainfalls(
    n_rainfalls, dt, range_A, params_B, range_D, dur_padding=60, multiplier=1
):
    rainfalls = []

    for i in range(n_rainfalls):
        rand_A = np.random.randint(range_A[0], range_A[1])
        rand_B = np.random.uniform() * (params_B[0]) + params_B[1]
        rand_dur = np.random.randint(range_D[0], range_D[1]) * dt

        random_idf = {"A": rand_A, "B": rand_B, "n": 1.09}

        # specs_rain = ["Synt_" + str(i), random_idf, rand_dur]
        one_random_rain = altblocks(
            random_idf,
            dur=rand_dur,
            dt=dt,
            dur_padding=dur_padding,
            multiplier=multiplier,
        )
        # rainfalls.append((specs_rain, one_random_rain))
        rainfalls.append(one_random_rain)

    return rainfalls


# Time - dates conversor
def conv_time(time):
    time /= 60
    hours = int(time)
    minutes = (time * 60) % 60
    # seconds = (time*3600) % 60
    return "%d:%02d" % (hours, minutes)


# Modified function. Original taken from https://pyhyd.blogspot.com/2017/07/alternating-block-hyetograph-method.html
def altblocks(idf, dur, dt, dur_padding=60, multiplier=1):

    value_padding_pre = np.zeros(int(5 / dt))
    value_padding = np.zeros(int(dur_padding / dt))

    aPad = np.arange(dt, 5 + dt, dt)
    aPostPad = np.arange(dur_padding + 5 + dt, 5 + dur_padding + dur + dt, dt)

    aDur = np.arange(dt, dur + dt, dt)  # in minutes
    aInt = idf["A"] / (
        aDur ** idf["n"] + idf["B"]
    )  # idf equation - in mm/h for a given return period
    aDeltaPmm = np.diff(
        np.append(0, np.multiply(aInt, aDur / 60.0))
    )  # Duration: min -> hours
    aOrd = np.append(
        np.arange(1, len(aDur) + 1, 2)[::-1], np.arange(2, len(aDur) + 1, 2)
    )

    prec = np.asarray([aDeltaPmm[x - 1] for x in aOrd])

    aDur = aDur + 5
    prec = np.concatenate((value_padding_pre, prec, value_padding)) * multiplier
    aDur = np.concatenate((aPad, aDur, aPostPad))

    prec_str = list(map(str, np.round(prec, 2)))

    aDur_time = list(map(conv_time, aDur))
    aDur_str = list(map(str, aDur_time))

    aAltBl = dict(zip(aDur_str, prec_str))

    return aAltBl


def rain_blocks(values, durations, dt):
    values = np.array(values)
    durations = np.array(durations)
    dur = np.sum(durations)
    aDur = np.arange(dt, dur + dt, dt)  # in minutes
    aDur_time = list(map(conv_time, aDur))

    repetitions = np.int_(durations / dt)

    prec = np.repeat(values, repetitions)
    prec_str = list(map(str, np.round(prec, 2)))
    aDur_str = list(map(str, aDur_time))
    ans = dict(zip(aDur_str, prec_str))
    return ans


def get_max_from_raindict(dict_rain):
    list_of_values = list(dict_rain.values())
    return np.array(list_of_values).max()


# Generator of inp-readable lines for the rainfalls
def new_rain_lines(rainfall_dict, name_new_rain="name_new_rain"):

    new_lines_rainfall = []

    for key, value in rainfall_dict.items():
        year = key[:4]
        month = key[4:6]
        day = key[6:8]
        date = (
            month + "/" + day + "/" + year
        )  # Month / day / year (because of the inp format)

        time = key[8:10] + ":" + key[10:12]

        new_lines_rainfall.append(
            tabulator([name_new_rain, date, time, str(value * 0.01)])
        )

    return new_lines_rainfall


# Formatter for rain text lines --------------------------------------------------------------------------------
def create_datfiles(rainfalls, rainfall_dats_directory, identifier, isReal, offset=0):
    for idx, single_event in enumerate(rainfalls):

        if isReal:
            string_rain = [
                "".join(i)
                for i in get_new_rain_lines_real(single_event, name_new_rain="R1")
            ]
        else:
            string_rain = [
                "".join(i) for i in new_rain_lines_dat(single_event, name_new_rain="R1")
            ]

        filename = identifier + "_" + str(idx + offset) + ".dat"
        with open(rainfall_dats_directory / filename, "w") as f:
            f.writelines(string_rain)


def new_rain_lines(rainfall_dict, name_new_rain="name_new_rain", day="1/1/2019"):

    new_lines_rainfall = []

    for key, value in rainfall_dict.items():
        new_lines_rainfall.append(
            tabulator([name_new_rain, day, key, str(value)])
        )  # STA01  2004  6  12  00  00  0.12

    return new_lines_rainfall


def new_rain_lines_dat(
    rainfall_dict, name_new_rain="name_new_rain", day="1", month="1", year="2019"
):

    new_lines_rainfall = []

    for key, value in rainfall_dict.items():
        hour, minute = key.split(":")
        new_lines_rainfall.append(
            tabulator(
                [
                    name_new_rain,
                    year,
                    month,
                    str(int(day) + int((int(hour) / 24) % 24)),
                    str(int(hour) % 24),
                    minute,
                    str(value),
                ]
            )
        )  # STA01  2004  6  12  00  00  0.12

    return new_lines_rainfall


def get_new_rain_lines_real(
    rainfall_dict, name_new_rain="name_new_rain"
):  # , day='1', month = '1', year = '2019'):

    new_lines_rainfall = []

    for key, value in rainfall_dict.items():
        year = key[:4]
        month = key[4:6]
        day = key[6:8]
        hour, minute = key[8:10], key[10:12]
        new_lines_rainfall.append(
            tabulator([name_new_rain, year, month, day, hour, minute, str(value)])
        )  # STA01  2004  6  12  00  00  0.12

    return new_lines_rainfall


def get_lines_from_textfile(path):
    with open(path, "r") as fh:
        lines_from_file = fh.readlines()
    return lines_from_file


def run_SWMM(inp_path, rainfall_dats_directory, simulations_path, padding_hours=12):
    """
    Run SWMM for each event in the rainfall_dats_directory
    The inp file is modified to include the start and end dates of the event.
    The dat file is copied to the simulation folder.
    The padding hours are added to the end date of the event.
    The output files are saved in the simulations_path
    args:
    inp_path: Path
        Path to the inp file
    rainfall_dats_directory: Path
        Path to the directory containing the dat files
    simulations_path: Path
        Path to the directory where the simulations will be saved
    padding_hours: int
        Number of hours to add to the end date of the event
    """

    list_of_rain_datfiles = os.listdir(rainfall_dats_directory)
    columns = ["event_name", "start_date", "end_date", "end_time", "simulation_time"]
    df_info = pd.DataFrame(columns=columns)

    for event in list_of_rain_datfiles:
        print("Running simulation for", event)
        rain_event_path = rainfall_dats_directory / event

        inp = get_lines_from_textfile(inp_path)
        dat = get_lines_from_textfile(rain_event_path)

        for ln, line in enumerate(inp):
            splitted_line_dat = dat[0].split("\t")
            new_date = "".join(
                [
                    splitted_line_dat[2],  # Month
                    "/",
                    splitted_line_dat[3],  # Day
                    "/",
                    splitted_line_dat[1],  # Year
                ]
            )

            splitted_line_dat_last = dat[-1].split("\t")
            new_last_date = "".join(
                [
                    splitted_line_dat_last[2],
                    "/",
                    splitted_line_dat_last[3],
                    "/",
                    splitted_line_dat_last[1],
                ]
            )

            new_last_time = "".join(
                [splitted_line_dat_last[4], ":", splitted_line_dat_last[5], ":", "00"]
            )

            new_last_date_time = "".join([new_last_date, " ", new_last_time])
            # Convert new_date_time into a datetime object
            date_time_obj = datetime.strptime(new_last_date_time, "%m/%d/%Y %H:%M:%S")

            # Add 12 hours
            date_time_obj += timedelta(hours=padding_hours)

            # Convert back to string
            new_last_date_time = date_time_obj.strftime("%m/%d/%Y %H:%M:%S")

            # Separate date from time
            new_last_date, new_last_time = new_last_date_time.split(" ")

            if "START_DATE" in line:
                inp[ln] = line.replace(line.split()[-1], new_date)

            elif "END_DATE" in line:
                inp[ln] = line.replace(line.split()[-1], new_last_date)

            elif "END_TIME" in line:
                inp[ln] = line.replace(line.split()[-1], new_last_time)

            elif "PLACEHOLDER1" in line:
                inp[ln] = line.replace(
                    "PLACEHOLDER1", str(rainfall_dats_directory / event)
                )

        nf = simulations_path / event.replace(".dat", "")

        if os.path.exists(nf):
            print(nf, "already exists")
            continue
        else:
            os.mkdir(nf)

            with open(nf / "model.inp", "w") as fh:
                for line in inp:
                    fh.write("%s" % line)
            with open(nf / event, "w") as fh:
                for line in dat:
                    fh.write("%s" % line)

            start_time = time.time()
            subprocess.run(
                [
                    swmm_executable_path,
                    nf / "model.inp",
                    nf / "model.rpt",
                    nf / "model.out",
                ]
            )
            simulation_time = time.time() - start_time

            df_event = pd.DataFrame(
                [[event, new_date, new_last_date, new_last_time, simulation_time]],
                columns=columns,
            )
            df_info = pd.concat([df_info, df_event])

            execution_times_path = simulations_path.parents[0]

            if not execution_times_path.exists():
                execution_times_path.mkdir(parents=True)

            df_info.to_excel(
                execution_times_path / f"execution_times_{simulations_path.name}.xlsx",
                sheet_name="Execution times",
            )


def get_time_duration_from_running_SWMM(
    inp_path, rainfall_dats_directory, simulations_path
):
    list_of_rain_datfiles = os.listdir(rainfall_dats_directory)

    columns = ["event_name", "start_date", "end_date", "end_time", "simulation_time"]

    df_info = pd.DataFrame(columns=columns)

    for event in list_of_rain_datfiles:
        rain_event_path = rainfall_dats_directory + "\\" + event

        inp = get_lines_from_textfile(inp_path)
        dat = get_lines_from_textfile(rain_event_path)
        splitted_line_dat = dat[0].split("\t")
        new_date = "".join(
            [splitted_line_dat[2], "/", splitted_line_dat[3], "/", splitted_line_dat[1]]
        )
        splitted_line_dat_last = dat[-1].split("\t")
        new_last_date = "".join(
            [
                splitted_line_dat_last[2],
                "/",
                splitted_line_dat_last[3],
                "/",
                splitted_line_dat_last[1],
            ]
        )

        new_last_time = "".join(
            [splitted_line_dat_last[4], ":", splitted_line_dat_last[5], ":", "00"]
        )

        for ln, line in enumerate(inp):
            if "START_DATE" in line:
                inp[ln] = line.replace(line.split()[-1], new_date)
            elif "END_DATE" in line:
                inp[ln] = line.replace(line.split()[-1], new_last_date)
            elif "END_TIME" in line:
                inp[ln] = line.replace(line.split()[-1], new_last_time)

            elif "PLACEHOLDER1" in line:
                inp[ln] = line.replace(
                    "PLACEHOLDER1", "\\".join((rainfall_dats_directory, event))
                )

        nf = "\\".join((simulations_path, event.replace(".dat", "")))
        os.mkdir(nf)

        with open(nf + "\\model.inp", "w") as fh:
            for line in inp:
                fh.write("%s" % line)

        start_time = time.time()
        subprocess.run(
            [
                r"C:\Program Files (x86)\EPA SWMM 5.1.015\swmm5.exe",
                nf + "\\model.inp",
                nf + "\\model.rpt",
                nf + "\\model.out",
            ]
        )
        simulation_time = time.time() - start_time

        df_event = pd.DataFrame(
            [[event, new_date, new_last_date, new_last_time, simulation_time]],
            columns=columns,
        )
        df_info = pd.concat([df_info, df_event])

    return df_info


def extract_SWMM_results(simulations_path):
    list_of_simulations = os.listdir(simulations_path)
    for sim in list_of_simulations:
        print("Extracting simulation", sim)
        c_simulation_folder_path = Path(simulations_path) / sim
        working_out = c_simulation_folder_path / "model.out"

        if not os.path.exists(c_simulation_folder_path / "flow_rate.csv"):
            head_out_timeseries = swm.extract(working_out, "node,,Hydraulic_head")
            runoff_timeseries = swm.extract(working_out, "subcatchment,,Runoff_rate")
            #            total_inflow_timeseries = swm.extract(working_out, "node,,Total_inflow")
            flow_rate_timeseries = swm.extract(working_out, "link,,Flow_rate")

            head_out_timeseries.to_csv(c_simulation_folder_path / "hydraulic_head.csv")
            runoff_timeseries.to_csv(c_simulation_folder_path / "runoff.csv")
            flow_rate_timeseries.to_csv(c_simulation_folder_path / "flow_rate.csv")
        else:
            print("Already extracted")


def extract_info_inp(lines, line_where, header, names=[], elevation=[]):
    offset = 3
    c_line = lines[line_where[header] + offset]

    while c_line != "\n":
        if c_line[0] != ";":
            names.append(c_line.split()[0])
            elevation.append(c_line.split()[1])
        offset += 1
        c_line = lines[line_where[header] + offset]
    return names, elevation
