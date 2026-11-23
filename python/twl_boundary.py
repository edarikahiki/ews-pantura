import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import hvplot.pandas
from scipy.spatial import cKDTree
import os
from pathlib import Path
from scipy import stats 
from scipy.signal import find_peaks, detrend
from scipy.optimize import curve_fit
from math import ceil, trunc
from sklearn.metrics import mean_squared_error
from python.HGRAPHER_functions_mod import *


def open_gtsm_subset(data_dir, query_xy, k=1, radius=None, chunks=None):

    fn =  os.listdir(data_dir)
    files = [os.path.join(data_dir,f) for f in fn]

    ds0 = xr.open_dataset(files[0], chunks=chunks)
    x = ds0["station_x_coordinate"].values
    y = ds0["station_y_coordinate"].values
    xy = np.column_stack([x, y])
    tree = cKDTree(xy)

    query_xy = np.asarray(query_xy)

    if radius is not None:
        hits = tree.query_ball_point(query_xy, r=radius)
        station_idx = np.unique(np.concatenate(hits))
    else:
        dist, idx = tree.query(query_xy, k=k)
        station_idx = np.unique(np.atleast_1d(idx).ravel())

    dss = [xr.open_dataset(f, chunks=chunks).isel(stations=station_idx) for f in files]
    ds = xr.concat(dss, dim="time")

    station_idx = []
    for i in idx:
        subset = ds0.isel(stations=np.atleast_1d(i).ravel())
        station_idx.append(subset['stations'].values)
    station_idx = np.array(station_idx)

    return ds, station_idx, dist


def interpolate_boundary_waterlevel(ds, idx, dist,key):

    wl_list = []
    for i in range(len(dist)):
        st_i = np.atleast_1d(idx[i])
        d_i = np.atleast_1d(dist[i])

        wl_i = ds[key].sel(stations=st_i)

        w = 1 / np.maximum(d_i, 1e-12)
        w = w / w.sum()

        # wl_interp = (w * wl_i.values).sum(axis=1)

        wl_interp = (wl_i * xr.DataArray(w, dims=["stations"])).sum("stations")
        wl_list.append(wl_interp)

    wl_boundary = xr.concat(wl_list, dim="boundary_point")
    wl_boundary = wl_boundary.transpose("time", "boundary_point")
    # wl_boundary = np.array(wl_list)
    return wl_boundary


def ecdf(var):
    x = np.sort(var)
    n = x.size
    y = np.arange(1, n + 1)/(n + 1)
    return [y, x]

def eva_POT(df, return_period, percentile, make_plot=True):
# using raw data
    x = df

    # # using rolling detrend
    # trend = df.rolling("60D", center=True).mean()
    # x = df - trend

    # print(type(x),np.shape(x))
    # x.values.squeeze()

    # # using scipy detrend
    # x = detrend(df[0])
    # x = pd.Series(x)
    # print(type(x),np.shape(x))
    # x.values.squeeze()

    nyear = df.index.year.nunique()

    # threshold = np.quantile(df,0.995,axis=0)
    # threshold = float(threshold)
    threshold = x.quantile(percentile)
    distance = 24*4 #days
    peaks, _ = find_peaks(x.values.squeeze(), height=threshold, distance=distance)
    npeaks = peaks.shape
    nevent = npeaks[0]/nyear

    t_POT = x.index[peaks]
    wl_POT = x.iloc[peaks]

    print(f'Threshold = {threshold} m')
    print(f'Number of peaks = {(npeaks[0]):.0f}')
    print(f'Number of event per year Ns = {(nevent):.2f}')


    # fit to GPD
    params_POT = stats.genpareto.fit(wl_POT - threshold, floc=0)
    return_period_range = np.linspace(0.1, 1000, 10000)
    return_period_range = np.round(return_period_range,1)

    Prob_POT = 1 - 1/(return_period_range*nevent)
    GPD_return_levels = stats.genpareto.ppf(Prob_POT, *params_POT) + threshold

    freq, cdf = ecdf(wl_POT.values.squeeze())

    # plt.scatter(y_data,y_GPD)
    RT_data = 1/((1 - freq)*nevent)
    Prob_data = 1 - 1/(RT_data*nevent)
    # plt.plot(Prob_data)

    y_GPD = stats.genpareto.ppf(Prob_data, *params_POT) + threshold
    y_data = cdf


    mse = mean_squared_error(y_data, y_GPD)
    rmse_GPD = np.sqrt(mse)
    print(f"RMSE GPD = {rmse_GPD:.3f}")

    # plt.scatter(y_data,y_GPD)
    
    RTs = [1, 10, 50, 100, 200, 500, 1000]

    indices = [np.abs(return_period_range - r).argmin() for r in RTs]

    table = pd.DataFrame({
        "RP": RTs,
        "WL": GPD_return_levels[indices]
    }).round(3)

    # # get snaphot with max value at center
    # maxloc = x.argmax()
    # snapshot = x[maxloc-24:maxloc+24]

    # # scale snaphot with extreme level
    # eva = table[table['RP'] == return_period]['WL'].values
    # eva_ts = eva/x.max()*(snapshot)


    # average_tide_signal, spring_tide_signal=generate_tide_signals(station,make_plot)
    # surge_hydrograph_height, surge_hydrograph_duration_mean = generate_surge_hydrograph(station,percentile,make_plot)
    # storm_tide_hydrograph_average_tide_signal,storm_tide_hydrograph_spring_tide_signal = generate_storm_tide_hydrograph(
    #     station,average_tide_signal,spring_tide_signal,surge_hydrograph_duration_mean,surge_hydrograph_height,
    #     percentile,rp,offset,make_plot)


    if make_plot:
        plt.figure(figsize=[20,7])
        plt.plot(x,alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Total Water Level(m)')
        plt.xlim([x.index[000],x.index[-1]])
        # filtered_df['swh'].iloc[peaks].plot(kind='scatter', color='red')#(kind='scatter')
        plt.scatter(t_POT,wl_POT,c='red',label='Peak Data')
        plt.title('Identified Peaks Using POT')
        plt.axhline(y = threshold,ls = 'dashed', c='black',lw=3,label ='Threshold')
        plt.legend()

        plt.figure(figsize=(11, 5))
        plt.plot(return_period_range, GPD_return_levels, 'cornflowerblue', label = 'GPD', linewidth=3, ls = '--')
        plt.xscale('log') 
        plt.scatter(1/((1 - freq)*nevent),cdf,c='red',s=8,label='Data from POT')
        plt.ylabel('Water Level [m]')
        plt.xlabel('RT [years]')
        plt.title('Return Level')
        plt.grid()
        plt.legend()

        # plt.figure(figsize=(10,3))
        # plt.plot(snapshot,label = 'Original signal')
        # plt.plot(eva_ts,label = 'Extreme value signal')
        # plt.legend()
        # plt.ylabel('Total Water Level [m]')
        # plt.title('Extreme event time series')

    return table#, eva_ts.values


def create_twl_boundary(xy,RP=100,
                        k=1,dt=600,nhour=36,
                        # data_dir = r'f:\TUD_CEG_Repository\MSc_Thesis\Data\CMIP6_total_level',
                        percentile = 0.999,
                        tide_type = 'average',
                        out_dir = os.getcwd(),
                        trim_data = True,
                        make_plot=False):

    
    # fn =  os.listdir(data_dir)
    # files = [os.path.join(data_dir,f) for f in fn]

    # boundary_xy = np.array([
    #     [3.890,53.923],
    #     # [106.80, -6.10],
    #     # [106.85, -6.15],
    #     # [106.90, -6.20],
    # ])

    # boundary_xy= np.array([[110.37639669,  -6.95031792],
    #     [110.39624847,  -6.93064921],
    #     [110.4263995 ,  -6.91749441],
    #     [110.45846793,  -6.90614794],
    #     [110.47915823,  -6.88108035]
    #     ])
    
    print('Querying TWL database.....')
    twl_data = r'f:\TUD_CEG_Repository\MSc_Thesis\Data\CMIP6_total_level'
    ds_twl, twl_station_idx, twl_dist = open_gtsm_subset(
        # np.atleast_1d(files[-1]),
        twl_data,
        xy,
        k=k,
        chunks={}
    )


    print('Interpolate TWL data to boundary point.....')
    twl_boundary = interpolate_boundary_waterlevel(ds_twl, twl_station_idx, twl_dist,'waterlevel')

    print('Querying Surge database.....')
    surge_data = r'f:\TUD_CEG_Repository\MSc_Thesis\Data\CMIP6_surge'
    ds_surge, surge_station_idx, surge_dist = open_gtsm_subset(
        # np.atleast_1d(files[-1]),
        surge_data,
        xy,
        k=k,
        chunks={}
    )

    print('Interpolate surge data to boundary point.....')
    surge_boundary = interpolate_boundary_waterlevel(ds_surge, surge_station_idx, surge_dist,'surge')

    # print('Calculate tidal time series.....')
    # tide_boundary = twl_boundary - surge_boundary
 
    print('Extracting time series.....')
    twl_df = twl_boundary.to_pandas()
    surge_df = surge_boundary.to_pandas()
    tide_df = twl_df - surge_df

    print('Resampling data.....')
    # tide_df = tide_boundary.to_pandas()
    tide_df = tide_df.asfreq(f"{dt}s").interpolate(method="time")
    surge_df = surge_df.asfreq(f"{dt}s").interpolate(method="time")


    #### correct this part before move on
    # dt = 3600
    # nhour = 48
    # t = np.arange(0,nhour*dt,dt)
    ts_average = []
    ts_spring = []
    for col in twl_df.keys():
    # for tidepd, surgepd in zip()    

        print(f'Extreme value analysis for boundary {col}')
        RPs = eva_POT(twl_df[col], RP, percentile, make_plot=make_plot)

        average_tide_signal, spring_tide_signal=generate_tide_signals(tide_df[col],make_plot=make_plot)
        surge_hydrograph_height, surge_hydrograph_duration_mean = generate_surge_hydrograph(surge_df[col],percentile,make_plot=make_plot)
        storm_tide_hydrograph_average_tide_signal,storm_tide_hydrograph_spring_tide_signal = generate_storm_tide_hydrograph(
            average_tide_signal,spring_tide_signal,surge_hydrograph_duration_mean,surge_hydrograph_height,
            RP,RPs,make_plot=make_plot)
        
        if trim_data:
            storm_tide_hydrograph_average_tide_signal = trim_timeseries(storm_tide_hydrograph_average_tide_signal,nhour=nhour,dt=dt)
            storm_tide_hydrograph_spring_tide_signal = trim_timeseries(storm_tide_hydrograph_average_tide_signal,nhour=nhour,dt=dt)


        ts_average.append(storm_tide_hydrograph_average_tide_signal)
        ts_spring.append(storm_tide_hydrograph_spring_tide_signal)

        # bzs.append(ts)

    # bzs = np.column_stack(bzs)
    # bzs = np.column_stack([t,bzs])
    if tide_type == 'average':
        bzs = np.column_stack(ts_average)
    elif tide_type == 'spring':
        bzs = np.column_stack(ts_spring)
    else:
        print('Wrong tine series type!!!!')
        return

    t = np.arange(0,bzs.shape[0]*dt,dt)
    bzs = np.column_stack([t,bzs])

    print('Saving bzs file.....')
    out_path = os.path.join(out_dir,'twl.bzs')
    np.savetxt(out_path,bzs,delimiter='\t',fmt="%.6f")

    return out_path

# def trim_timeseries(signal,nhour=72,dt=600,check_data=False):

#     ts = pd.Series(signal)
#     ts.index = ts.index*dt

#     peak = ts.argmax()
#     windows = int(nhour*(3600/dt)/2) #convert to second
#     rise = ts[:peak]
#     fall = ts[peak:]


#     rise_cross = np.where(np.sign(rise.shift(1)*rise)<0)
#     fall_cross = np.where(np.sign(fall.shift(1)*fall)<0)

#     rise_mask = rise.iloc[rise_cross].index > rise[-windows:].index[0]
#     first_idx = rise.iloc[rise_cross][rise_mask].index[0]

#     fall_mask = fall.iloc[fall_cross].index < fall[:windows].index[-1]
#     last_idx = fall.iloc[fall_cross][fall_mask].index[-1]

#     trimmed_ts = ts.copy()
#     trimmed_ts.loc[:first_idx] = np.nan
#     trimmed_ts.loc[last_idx:] = np.nan

#     if check_data:
#         rise.plot()
#         ax = plt.gca()
#         fall.plot(ax=ax)

#         # if rise_idx is not None:
#         ax.plot(first_idx, ts.loc[first_idx], 'o', label='rise crossing')

#         # if fall_idx is not None:
#         ax.plot(last_idx, ts.loc[last_idx], 'o', label='fall crossing')

#         # window lines (your original)
#         ax.axvline(ts.index[peak-int(nhour*6/2)])
#         ax.axvline(ts.index[peak+int(nhour*6/2)])

#         trimmed_ts.plot(ax=ax,label='trimmed',marker='x')

#         plt.grid()
#         plt.legend()
#         plt.show()

#     return trimmed_ts.values

def trim_timeseries(signal, nhour=72, dt=600, check_data=False):
    ts = pd.Series(signal, dtype="float64")
    ts.index = ts.index * dt  # seconds

    if ts.empty or ts.notna().sum() == 0:
        return ts.values

    peak_pos = ts.values.argmax()
    peak_time = ts.index[peak_pos]

    half_window_steps = int(nhour * 3600 / dt / 2)

    start_pos = max(0, peak_pos - half_window_steps)
    stop_pos  = min(len(ts) - 1, peak_pos + half_window_steps)

    rise = ts.iloc[:peak_pos + 1]
    fall = ts.iloc[peak_pos:]

    # zero-crossings where sign changes
    rise_cross_pos = np.where((rise.shift(1) * rise) < 0)[0]
    fall_cross_pos = np.where((fall.shift(1) * fall) < 0)[0]

    # keep only crossings inside allowed window
    rise_cross_pos = rise_cross_pos[rise_cross_pos >= start_pos]

    # fall is sliced from peak_pos, so convert local fall positions to global positions
    fall_cross_global = fall_cross_pos + peak_pos
    fall_cross_global = fall_cross_global[fall_cross_global <= stop_pos]

    # safe fallback if no crossing found
    if len(rise_cross_pos) > 0:
        first_pos = rise_cross_pos[0]
    else:
        first_pos = start_pos

    if len(fall_cross_global) > 0:
        last_pos = fall_cross_global[-1]
    else:
        last_pos = stop_pos

    trimmed_ts = ts.copy()
    trimmed_ts.iloc[:first_pos] = np.nan
    trimmed_ts.iloc[last_pos + 1:] = np.nan

    if check_data:
        ax = ts.plot(label="original", figsize=(10, 4))

        ax.plot(ts.index[first_pos], ts.iloc[first_pos], "o", label="start")
        ax.plot(ts.index[last_pos], ts.iloc[last_pos], "o", label="stop")

        ax.axvline(ts.index[start_pos], linestyle=":", label="window start")
        ax.axvline(ts.index[stop_pos], linestyle="--", label="window stop")

        trimmed_ts.plot(ax=ax, label="trimmed", marker="x")

        ax.grid()
        ax.legend()
        plt.show()

    return trimmed_ts.values