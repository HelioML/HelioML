import numpy as np
import datetime
import pandas as pd

__author__ = "James Paul Mason"
__contact__ = "jmason86@gmail.com"


def sod_to_hhmmss(sod):
    """Convert seconds of day to 'hh:mm:ss'.

    Inputs:
        sod [np.array]: The array of seconds of day to convert.

    Optional Inputs:
        None.

    Outputs:
        hhmmss [np chararray]: Array of strings of the form 'hh:mm:ss'.

    Optional Outputs:
        None.

    Example:
        hhmmss = sod_to_hhmmss(sod)
    """

    # Parse seconds of day into hours, minutes, and seconds
    time = sod % (24 * 3600)
    time = time.astype(int)
    hours = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time

    # Convert to string type and add leading zeros
    hours_str = hours.astype(str)
    small_hour_indices = hours < 10
    hours_str[small_hour_indices] = np.core.defchararray.zfill(hours_str[small_hour_indices], 2)

    minutes_str = minutes.astype(str)
    small_minute_indices = minutes < 10
    minutes_str[small_minute_indices] = np.core.defchararray.zfill(minutes_str[small_minute_indices], 2)

    seconds_str = seconds.astype(str)
    small_second_indices = seconds < 10
    seconds_str[small_second_indices] = np.core.defchararray.zfill(seconds_str[small_second_indices], 2)

    return np.char.array(hours_str) + ':' + np.char.array(minutes_str) + ':' + np.char.array(seconds_str)


def yyyydoy_to_datetime(yyyydoy):
    """Convert yyyydoy (e.g., 2017013) date to datetime (e.g., datetime.datetime(2017, 1, 13, 0, 0)).

    Inputs:
        yyyydoy [np.array]: The array of dates to convert.

    Optional Inputs:
        None.

    Outputs:
        dt_array [np.array]: Array of datetimes.

    Optional Outputs:
        None.

    Example:
        dt_array = yyyydoy_to_yyyymmdd(yyyydoy)
    """

    # Parse year and doy
    parsed_date = np.modf(yyyydoy / 1000)  # Divide to get yyyy.doy

    # Convert to array of yyyymmdd datetimes and return
    return np.array([datetime.datetime(int(parsed_date[1][i]), 1, 1) +           # base year (yyyy-01-01)
                     datetime.timedelta(days=int(parsed_date[0][i] * 1000) - 1)  # doy -> mm-dd
                     for i in range(len(yyyydoy))])                              # loop over input array


def yyyydoy_sod_to_datetime(yyyydoy, sod):
    """Convert yyyydoy, sod (e.g., 2017013, 86395) to datetime (e.g., datetime.datetime(2017, 1, 13, 23, 59, 55)).

    Inputs:
        yyyydoy [np.array]: The array of dates to convert.
        sod [np.array]:     The array of seconds of day to convert.

    Optional Inputs:
        None.

    Outputs:
        dt_array [np.array]: Array of datetimes.

    Optional Outputs:
        None.

    Example:
        dt_array = yyyydoy_sod_to_datetime(yyyydoy, sod)
    """

    # Parse year and doy
    parsed_date = np.modf(yyyydoy / 1000)  # Divide to get yyyy.doy

    # Convert sod to hhmmss
    hhmmss = sod_to_hhmmss(sod)

    return np.array([datetime.datetime(int(parsed_date[1][i]), 1, 1,
                     int(hhmmss[i][:2]),
                     int(hhmmss[i][3:5]),
                     int(hhmmss[i][6:8])) +                                      # base (yyyy-01-01 hh:mm:ss)
                     datetime.timedelta(days=int(parsed_date[0][i] * 1000) - 1)  # doy -> mm-dd
                     for i in range(len(yyyydoy))])                              # loop over input array


def metatimes_to_seconds_since_start(metatimes):
    """Convert metatime array to seconds since first datetime in array.

    Inputs:
        metatimes [np.array or pd.DatetimeIndex]: Array of metatimes.
                                                  metatimes are either:
                                                  np.array of datetime.datetime,
                                                  np.array of np.datetime64,
                                                  np.array of pd.Timestamp,
                                                  pd.DatetimeIndex.

    Optional Inputs:
        None.

    Outputs:
        seconds_since_start [np.array]: Array of long integers
                                        indicating number of seconds since the first time in the arrray.

    Optional Outputs:
        None.

    Example:
        seconds_since_start = metatimes_to_seconds_since_start(metatimes)
    """

    # Check type of input and do conversion accordingly
    if isinstance(metatimes[0], datetime.datetime):
        dt_since_start = metatimes - metatimes[0]
        return np.array([np.long(metatime.total_seconds())
                        for metatime in dt_since_start])
    elif isinstance(metatimes[0], np.datetime64 or pd.Timestamp):
        npdts_since_start = metatimes - metatimes[0]
        return np.array([np.long(npdt_since_start / np.timedelta64(1, 's'))
                        for npdt_since_start in npdts_since_start])


def metatimes_to_human(metatimes):
    """Convert metatime array to an ISO string with the T and Z removed (yyyy-mm-dd hh:mm:ss).

    Inputs:
        metatimes [np.array or pd.DatetimeIndex]: Array of metatimes.
                                                  metatimes are either:
                                                  np.array of datetime.datetime,
                                                  np.array of np.datetime64,
                                                  np.array of pd.Timestamp,
                                                  pd.DatetimeIndex.

    Optional Inputs:
        None.

    Outputs:
        times_human [np.array]: Array of string format ISO times with the T and Z removed (yyyy-mm-dd hh:mm:ss).

    Optional Outputs:
        None.

    Example:
        times_human = metatimes_to_human(metatimes)
    """

    # Check type of input and do conversion accordingly
    if isinstance(metatimes[0], datetime.datetime):
        return np.array([metatime.strftime('%Y-%m-%d %H:%M:%S')
                        for metatime in metatimes])
    elif isinstance(metatimes[0], np.datetime64):
        return np.array([metatime.astype(str)
                        for metatime in metatimes])
    elif isinstance(metatimes[0], pd.Timestamp):
        return np.array([metatime.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
                        for metatime in metatimes])
    elif isinstance(metatimes[0], str):
        return metatimes


def seconds_since_start_to_metatimes(seconds_since_start, start_metatime):
    """Convert seconds since start to metatime with type identical to start_metatime.

    Inputs:
        seconds_since_start [np.array]: Array of long integers
                                        indicating number of seconds since the first time in the arrray.
        start_metatime [metatime]:      Starting time. Not an np.array, but otherwise follows the
                                        same available formats for metatime (see metatimes output description).

    Optional Inputs:
        None.

    Outputs:
        metatimes [np.array or pd.DatetimeIndex]: Array of times in the same format as start_metatime.
                                                  metatimes are either:
                                                  np.array of datetime.datetime,
                                                  np.array of np.datetime64,
                                                  np.array of pd.Timestamp,
                                                  pd.DatetimeIndex.

    Optional Outputs:
        None.

    Example:
        datetimeindex = seconds_since_start_to_metatimes(seconds_since_start, dt.datetime(2017, 5, 3, 13, 45, 49))
    """

    # Check type optional input and do conversion accordingly
    if isinstance(start_metatime, datetime.datetime):
        return np.array([start_metatime + datetime.timedelta(seconds=int(second_since_start))
                        for second_since_start in seconds_since_start])
    elif isinstance(start_metatime, np.datetime64):
        return np.array([start_metatime + np.timedelta64(int(second_since_start), 's')
                        for second_since_start in seconds_since_start])
    elif isinstance(start_metatime, pd.Timestamp):
        return np.array([pd.Timestamp(start_metatime + datetime.timedelta(seconds=int(second_since_start)))
                        for second_since_start in seconds_since_start])
    elif isinstance(start_metatime, pd.DatetimeIndex):
        return pd.DatetimeIndex(seconds_since_start_to_metatimes(seconds_since_start, start_metatime.values[0]))


def datetimeindex_to_iso(datetimeindex):
    """Convert pandas DatatimeIndex to an ISO string.

    Inputs:
        DatetimeIndex [pd.DatetimeIndex]: A pandas DatetimeIndex to be converted.

    Optional Inputs:
        None.

    Outputs:
        iso [np.array]: Array of string format ISO times.

    Optional Outputs:
        None.

    Example:
        iso = datetimeindex_to_iso(df.index)
    """
    return datetimeindex.strftime('%Y-%m-%dT%H:%m:%SZ')


def datetimeindex_to_human(datetimeindex):
    """Convert pandas DatatimeIndex to an ISO string with the T and Z removed (yyyy-mm-dd hh:mm:ss).

    Inputs:
        DatetimeIndex [pd.DatetimeIndex]: A pandas DatetimeIndex to be converted.

    Optional Inputs:
        None.

    Outputs:
        times_human [np.array]: Array of string format ISO times with the T and Z removed (yyyy-mm-dd hh:mm:ss).

    Optional Outputs:
        None.

    Example:
        times_human = datetimeindex_to_human(df.index)
    """
    return datetimeindex.strftime('%Y-%m-%d %H:%m:%S')
