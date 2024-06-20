# import packages
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime
import pyproj
from pyproj import transform
import functions_hiwi as fct


def sum_curve(data, station, date_start, date_end):

    # set DataFrame for this run of this funktion
    dataframe = data[station][date_start:date_end]
    
    # Messwerte der Station aufsummieren
    sum_list = dataframe.cumsum().tolist()
    sum = data[station][date_start:date_end].sum()

    # "Summendataframe" erstellen
    index_sum_df = dataframe.index
    sum_df = pd.Series(data=sum_list, index=index_sum_df)

    # Subplot erstellen
    fig, ax = plt.subplots()

    # plot Tageswerte
    ax.plot(index_sum_df, dataframe, label='Tageswerte', color='green')
    ax.set_ylabel('Tageswert [mm]')
    plt.xticks(rotation=45)
    plt.legend(loc=9)

    # plot Summenkurve
    ax = ax.twinx()
    ax.plot(index_sum_df, sum_list, label='Summenkurve', color='red')
    ax.set_ylabel('Summenkurve [mm]')
    plt.legend(loc=2)

    name_plot = 'Summenkurve + Tageswerte, Station: ' + str(station)
    plt.title(name_plot)
    plt.xlabel('DateTime')
    plt.legend()
    
    plt.show()
    plt.close()
    
    return print('Gesamtniederschlag über Zeitraum:', round(sum, 2), 'mm\n') # (print('Index Station', str(station), ': \n\n', index_sum_df, '\n\n', 'Summe aktuel zu Zeitstempel:\n\n', sum_df))

def nan_nonan_ratio(data, station):

    dataframe = data[station]
    
    nan_count = dataframe.isna().sum()
    ratio = (nan_count / len(dataframe))*100
    print('count of nans:', nan_count, '\nlength of data:', len(dataframe), '\nRatio of nan/nonan:', round(ratio, 2), '%')

    return

def longest_nan_sequence(data, station):

    dataframe = data[station].isna()
    
    longest_sequences = {}
    
    longest_sequence_length = 0
    current_sequence_length = 0
      
    for value in dataframe:
        if value == True:  # Wenn der Wert NaN ist
            current_sequence_length += 1
            if current_sequence_length > longest_sequence_length:
                longest_sequence_length = current_sequence_length
        else:
            current_sequence_length = 0
        
    longest_sequences[station] = longest_sequence_length
    
    return longest_sequences

def find_nan_sequence(data, station, day_start, day_end, max_nans):
    # 17*nan = 1h

    dataframe = data[station][day_start : day_end]
    position_in_dataframe = 0
    count = 0
    
    for value in dataframe:
        if np.isnan(value) == True:
            if count == 0:
                interval_start = dataframe.index[position_in_dataframe]
                
            count += 1
            if count == max_nans:
                interval_end = dataframe.index[position_in_dataframe]
                print('nan sequence of', max_nans, 'nans', '\ntime interval:', interval_start, ' to', interval_end)
                count = 0
            position_in_dataframe += 1
        else:
            position_in_dataframe += 1
            count = 0
            continue
            
    return

def i_nans_before_peak(data, y, station, quantile):

    if y == 'pr':
        timegap = datetime.timedelta(hours=1)
    elif y == 'sc':
        timegap = datetime.timedelta(minutes=5)

    dataframe = data[station]

    peaks = dataframe[dataframe > dataframe.quantile(quantile)]
    
    for index_peak in peaks.index:
        count = 0
        for i in reversed(dataframe.loc[: index_peak - timegap].isna()):
            if i == True:
                count += 1
            else:
                if count > 0:
                    print(count, 'leading nans before', index_peak)
                    break
                else:
                    # print('no leading nans before', index_peak)
                    break      
    return

def coordinates(loc_prim, loc_sec, y, station, ref1, ref2, ref3, ref4):
    
    if y == 'primary':
        coords_lon = loc_prim['lon']
        coords_lat = loc_prim['lat']
    elif y == 'secondary':
        coords_lon = loc_sec['lon']
        coords_lat = loc_sec['lat']
    elif y == 'both':
        coords_lon_prim = loc_prim['lon']
        coords_lat_prim = loc_prim['lat']
        coords_lon_sec = loc_sec['lon']
        coords_lat_sec = loc_sec['lat']

    if y == 'both':
        name_plot = 'Coordinates ' + y + ' networks'
        plt.scatter(x=coords_lon_prim, y=coords_lat_prim, s=20, color='red', label='primary network', marker='x', linewidth=1)
        plt.scatter(x=coords_lon_sec, y=coords_lat_sec, s=2, color='blue', label='secondary network', alpha=0.5)
        if type(station) == int:
            plt.scatter(loc_prim['lon'].iloc[station], loc_prim['lat'].iloc[station], color='black')
        plt.legend()
    else:
        name_plot = 'Coordinates ' + y + ' network'
        plt.scatter(x=coords_lon, y=coords_lat, s=10)
        if type(station) == int:
            if y == 'primary':
                plt.scatter(loc_prim['lon'].iloc[station], loc_prim['lat'].iloc[station], color='red')
            elif y == 'secondary':
                plt.scatter(loc_sec['lon'].iloc[station - 1], loc_sec['lat'].iloc[station - 1], color='red')

            try:
                plt.scatter(loc_sec['lon'].iloc[ref1 - 1], loc_sec['lat'].iloc[ref1 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref2 - 1], loc_sec['lat'].iloc[ref2 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref3 - 1], loc_sec['lat'].iloc[ref3 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref4 - 1], loc_sec['lat'].iloc[ref4 - 1], color='lime', s=10)
            except:
                pass

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(name_plot)

    plt.show()
    plt.close()
    
    return # print(coords_lon, coords_lat)

def find_primary_stations(loc_prim, lon_x1, lon_x2, lat_y1, lat_y2):
    
    for i in range(len(loc_prim)):
        lon = loc_prim['lon'][i]
        lat = loc_prim['lat'][i]
        if (lon <= lon_x2 and lon >= lon_x1) and (lat <= lat_y2 and lat >= lat_y1):
            print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
    
    return

def get_data_nan_seq_before_peak(data, y, station, quantile):

    '''
    outside of function, write:
    output_list_counts_start, output_list_counts_end, output_list_counts, output_list_index_peak = get_data_nan_seq_before_peak(...)
    '''

    if y == 'pr':
        timegap = datetime.timedelta(hours=1)
    elif y == 'sc':
        timegap = datetime.timedelta(minutes=5)

    dataframe = data[station]

    peaks = dataframe[dataframe > dataframe.quantile(quantile)]
    list_index_peak = []
    list_counts = []
    list_counts_start = []
    list_counts_end = []

    for index_peak in peaks.index:
        count = 0
        for i in reversed(dataframe.loc[: index_peak - timegap].isna()):
            if i == True:
                count += 1
            else:
                if count > 0:
                    count_start = index_peak - (timegap * count)
                    list_counts_start.append(count_start)
                    list_counts_end.append(index_peak - timegap)
                    list_counts.append(count)
                    list_index_peak.append(index_peak)
                    break
                else:
                    break      
    return list_counts_start, list_counts_end, list_counts, list_index_peak

def correct_data(data, reference, y, station, quantile):

    fct.get_data_nan_seq_before_peak(data, y, station, quantile) # to get the output lists
    output_list_counts_start, output_list_counts_end, output_list_counts, output_list_index_peak = fct.get_data_nan_seq_before_peak(data, y, station, quantile)
   
    data_corrected = data[[station]].copy() # copy the data to a new dataframe

    if y == 'pr':
        frequency = '1h'
    elif y == 'sc':
        frequency = '5min'

    for i in range(len(output_list_index_peak)):

        datetime_index = pd.date_range(start=output_list_counts_start[i], end=output_list_index_peak[i], freq=frequency) # create a datetime index for the time period of the nan sequence before the peak
        sum = reference[station].loc[output_list_counts_start[i] : output_list_index_peak[i]].sum() # sum of the reference values for the time period of the nan sequence before the peak
        value_peak = data[station].loc[output_list_index_peak[i]] # value of the peak

        for index in datetime_index:
            try:
                peak_portion = round(((reference[station].loc[index] / sum) * value_peak), 2)
            except ZeroDivisionError:
                peak_portion = 0
                
            data_corrected[station].loc[index] = peak_portion # replace the nan values with the calculated peak portion
        
    return data_corrected

def find_4_nearest_reference_stations(coordinates, station):
    
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    # set frame for search
    max_distance_of_reference_stations_lon = 5000
    max_distance_of_reference_stations_lat = 5000

    # set coordinates of the station
    lon_station = coordinates['lon'].iloc[station - 1]
    lat_station = coordinates['lat'].iloc[station - 1]

    list_reference_stations_lon = []
    list_reference_stations_lat = []
    list_station = []
    list_distance = []

    # find the 4 nearest stations in frame
    for i in range(len(coordinates)):
        lon = coordinates['lon'].iloc[i]
        lat = coordinates['lat'].iloc[i]
        if (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= (lon_station - max_distance_of_reference_stations_lon)) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= (lat_station - max_distance_of_reference_stations_lat)):
            if lon == lon_station and lat == lat_station:
                pass
            else:
                # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                # print('\n')

                distance = np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2)

                list_distance.append(distance)
                list_station.append(i + 1)
                list_reference_stations_lon.append(lon)
                list_reference_stations_lat.append(lat)
        
        array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
        df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

# die 4 nächstgelegenen Stationen herausfiltern
        
    if len(df_reference_stations) == 0:
        print('No reference stations found')
        pass
    elif len(df_reference_stations) <= 4:
        for i in range(len(df_reference_stations)):    
            # print(distance)
            pass
    elif len(df_reference_stations) > 4:
        
        list_distance_nearest_reference_stations = []

        for i in range(len(list_distance)):
            if len(list_distance_nearest_reference_stations) < 4:
                index_min_distance = np.argmin(list_distance)
                list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                del list_distance[index_min_distance]

                
        # print(list_distance_nearest_reference_stations)

        for i in df_reference_stations['distance']:
            count = 0
            for j in list_distance_nearest_reference_stations:
                if i == j:
                    count += 1
                    break
                else:
                    continue
            if count == 0:
                df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)

    return df_reference_stations

def find_reference_stations_for_each_quadrant(coordinates, station):
    
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    max_distance_of_reference_stations_lon = 10000000000
    max_distance_of_reference_stations_lat = 10000000000

    lon_station = coordinates['lon'].iloc[station - 1]
    lat_station = coordinates['lat'].iloc[station - 1]

    df_reference_stations_u_l = pd.DataFrame()
    df_reference_stations_u_r = pd.DataFrame()
    df_reference_stations_d_l = pd.DataFrame()
    df_reference_stations_d_r = pd.DataFrame()

    list_index = []
    list_distance_nearest = []

# für jeden Quadranten die jeweils nächste Station finden
    for k in range(4):
    
        list_reference_stations_lon = []
        list_reference_stations_lat = []
        list_station = []
        list_distance = []
        
        for i in range(len(coordinates)):
            lon = coordinates['lon'].iloc[i]
            lat = coordinates['lat'].iloc[i]

            # Quadranten definieren
            if k == 0:
                quadrant = (lon >= (lon_station - max_distance_of_reference_stations_lon) and lon <= lon_station) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= lat_station)
            elif k == 1:
                quadrant = (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= lon_station) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= lat_station)
            elif k == 2:
                quadrant = (lon >= (lon_station - max_distance_of_reference_stations_lon) and lon <= lon_station) and (lat >= (lat_station - max_distance_of_reference_stations_lat) and lat <= lat_station)
            elif k == 3:
                quadrant = (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= lon_station) and (lat >= (lat_station - max_distance_of_reference_stations_lat) and lat <= lat_station)
            
            if quadrant:
                if lon == lon_station and lat == lat_station:
                    pass
                else:
                    # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                    # print('\n')

                    distance = np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2)

                    list_distance.append(distance)
                    list_station.append('ams' + str(i + 1))
                    list_reference_stations_lon.append(lon)
                    list_reference_stations_lat.append(lat)
            
            array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
            df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

        # die nächstgelegene Station herausfiltern
            
        if len(df_reference_stations) == 0:
            # print('No reference stations found')
            pass
        elif len(df_reference_stations) == 1:
            pass
        elif len(df_reference_stations) > 1:
            
            list_distance_nearest_reference_stations = []

            # für jeden Quadranten die jeweils nächste Station finden
            for i in range(len(list_distance)):
                if len(list_distance_nearest_reference_stations) < 1:
                    index_min_distance = np.argmin(list_distance)
                    list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                    del list_distance[index_min_distance]

                    
            # print(list_distance_nearest_reference_stations)

            # df anpassen, sodass nur die nächstgelegene Station übrig bleibt
            for i in df_reference_stations['distance']:
                count = 0
                for j in list_distance_nearest_reference_stations:
                    if i == j:
                        count += 1
                        break
                    else:
                        continue
                if count == 0:
                    df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)
            
            if k == 0:
                df_reference_stations_u_l = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 1:
                df_reference_stations_u_r = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 2:
                df_reference_stations_d_l = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 3:
                df_reference_stations_d_r = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
                
            
            # print(df_reference_stations)
        df_reference_stations = pd.concat([df_reference_stations_u_l, df_reference_stations_u_r, df_reference_stations_d_l, df_reference_stations_d_r])        
            
    return list_index, list_distance_nearest, df_reference_stations

def berechnung_Gewichte(list_index, list_distance_nearest):
    
    list_weights = []

    if len(list_index) == 0:
        W1 = 0
        W2 = 0
        W3 = 0
        W4 = 0
    elif len(list_index) == 1:
        distance1 = list_distance_nearest[0]

        W1 = 1
        list_weights.append(W1)
    elif len(list_index) == 2:
        distance1 = list_distance_nearest[0]
        distance2 = list_distance_nearest[1]

        W1 = (1/distance1**2)/((1/distance1**2)+(1/distance2**2))
        W2 = (1/distance2**2)/((1/distance1**2)+(1/distance2**2))
        list_weights.append(W1)
        list_weights.append(W2)
    elif len(list_index) == 3:
        distance1 = list_distance_nearest[0]
        distance2 = list_distance_nearest[1]
        distance3 = list_distance_nearest[2]

        W1 = (1/distance1**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2))
        W2 = (1/distance2**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2))
        W3 = (1/distance3**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2))
        list_weights.append(W1)
        list_weights.append(W2)
        list_weights.append(W3)
    elif len(list_index) == 4:
        distance1 = list_distance_nearest[0]
        distance2 = list_distance_nearest[1]
        distance3 = list_distance_nearest[2]
        distance4 = list_distance_nearest[3]

        W1 = (1/distance1**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
        W2 = (1/distance2**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
        W3 = (1/distance3**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
        W4 = (1/distance4**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
        list_weights.append(W1)
        list_weights.append(W2)
        list_weights.append(W3)
        list_weights.append(W4)

    return list_weights

def berechnung_Referenzniederschlag(secondary_data_df_nonan, df_reference_values, list_weights, list_index):
    
    if list_weights == []:
        pass
    elif len(list_weights) == 1:
        for index in secondary_data_df_nonan.index:
            h1 = secondary_data_df_nonan[list_index[0]].loc[index]

            h_ref = list_weights[0]*h1

            df_reference_values.loc[index] = h_ref
    elif len(list_weights) == 2:
        for index in secondary_data_df_nonan.index:
            h1 = secondary_data_df_nonan[list_index[0]].loc[index]
            h2 = secondary_data_df_nonan[list_index[1]].loc[index]

            h_ref = list_weights[0]*h1 + list_weights[1]*h2

            df_reference_values.loc[index] = h_ref
    elif len(list_weights) == 3:
        for index in secondary_data_df_nonan.index:
            h1 = secondary_data_df_nonan[list_index[0]].loc[index]
            h2 = secondary_data_df_nonan[list_index[1]].loc[index]
            h3 = secondary_data_df_nonan[list_index[2]].loc[index]

            h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3

            df_reference_values.loc[index] = h_ref
    elif len(list_weights) == 4:
        for index in secondary_data_df_nonan.index:
            h1 = secondary_data_df_nonan[list_index[0]].loc[index]
            h2 = secondary_data_df_nonan[list_index[1]].loc[index]
            h3 = secondary_data_df_nonan[list_index[2]].loc[index]
            h4 = secondary_data_df_nonan[list_index[3]].loc[index]

            h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3 + list_weights[3]*h4

            df_reference_values.loc[index] = h_ref

    return df_reference_values

def calculate_reference_df(data_prim):
    index = pd.date_range(start='2019-12-31 23:05:00', end='2023-10-14 21:00:00', freq='5min')
    df_reference = pd.DataFrame(index=index, columns=['station'])
    df_reference

    for index in df_reference.index:
    
        try:
            h1 = data_prim['rr07'].loc[index]
            h2 = data_prim['rr10'].loc[index]

            h_ref = 0.5*h1 + 0.5*h2

            df_reference.loc[index] = h_ref
        except KeyError:
            df_reference.loc[index] = np.nan
    return df_reference

def correct_data_lauchaecker(data, df_reference, station, quantile):
        
    df_reference = df_reference.rename(columns={df_reference.columns[0] : station})
    data_corrected = fct.correct_data(data, df_reference, 'sc', station, quantile)

    return data_corrected

def LatLon_To_XY(i_area, j_area):
    
    ''' convert coordinates from wgs84 to utm 32'''
    P = pyproj.Proj(proj='utm', zone=32,
                    ellps='WGS84',
                    preserve_units=True)

    x, y = P.transform(i_area, j_area)

    return x, y

def resampleDf(df, agg, closed='right', label='right',
               shift=False, leave_nan=True,
               label_shift=None,
               temp_shift=0,
               max_nan=0):

    if shift == True:
        df_copy = df.copy()
        if agg != 'D' and agg != '1440min':
            raise Exception('Shift can only be applied to daily aggregations')
        df = df.shift(-6, 'H')

    # To respect the nan values
    if leave_nan == True:
        # for max_nan == 0, the code runs faster if implemented as follows
        if max_nan == 0:
            # print('Resampling')
            # Fill the nan values with values very great negative values and later
            # get the out again, if the sum is still negative
            df = df.fillna(-100000000000.)
            df_agg = df.resample(agg,
                                 closed=closed,
                                 label=label,
                                 offset=temp_shift).sum()
            # Replace negative values with nan values
            df_agg.values[df_agg.values[:] < 0.] = np.nan
        else:
            df_agg = df.resample(rule=agg,
                                 closed=closed,
                                 label=label,
                                 offset=temp_shift).sum()
            # find data with nan in original aggregation
            g_agg = df.groupby(pd.Grouper(freq=agg,
                                          closed=closed,
                                          label=label))
            n_nan_agg = g_agg.aggregate(lambda x: pd.isnull(x).sum())

            # set aggregated data to nan if more than max_nan values occur in the
            # data to be aggregated
            filter_nan = (n_nan_agg > max_nan)
            df_agg[filter_nan] = np.nan

    elif leave_nan == False:
        df_agg = df.resample(agg,
                             closed=closed,
                             label=label).sum()
    if shift == True:
        df = df_copy
    return df_agg

def calc_indicator_correlation(a_dataset, b_dataset, prob):
    """
    Tcalcualte indicator correlation two datasets

    Parameters
    ----------
    a_dataset: first data vector
    b_dataset: second data vector
    perc: percentile threshold 
    
    Returns
    ----------
    indicator correlation value

    Raises
    ----------

    """
    a_sort = np.sort(a_dataset)
    b_sort = np.sort(b_dataset)
    ix = int(a_dataset.shape[0] * prob)
    a_dataset[a_dataset < a_sort[ix]] = 0 # a_sort[ix] liefert perzentilwert abhängig von prob
    b_dataset[b_dataset < b_sort[ix]] = 0
    a_dataset[a_dataset > 0] = 1
    b_dataset[b_dataset > 0] = 1
    cc = np.corrcoef(a_dataset, b_dataset)[0, 1]

    return cc, a_dataset, b_dataset

def calculate_correlation_with_resample(data_primary, data_secondary, prim_station, sec_station, percentile):
    df_agg = fct.resampleDf(data_secondary[sec_station], 'H', closed='right', label='right', shift=False, leave_nan=True, label_shift=None, temp_shift=0, max_nan=2)

    index_start = df_agg.index[0]
    index_end = df_agg.index[-1]

    df_reference = data_primary[prim_station][index_start:index_end]

    df_for_correlation = pd.concat([df_reference, df_agg], axis=1)
    df_for_correlation = df_for_correlation.dropna() # ohne wäre correlation nan

    cc, a_dataset, b_dataset = fct.calc_indicator_correlation(df_for_correlation.iloc[:, 0], df_for_correlation.iloc[:, 1], percentile)

    return cc

def find_reference_stations_for_each_quadrant_with_plot_and_primary_station(loc_prim, loc_sec, station):
    
    coordinates = loc_sec
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    max_distance_of_reference_stations_lon = 3000
    max_distance_of_reference_stations_lat = 3000

    lon_station = coordinates['lon'].iloc[station - 1]
    lat_station = coordinates['lat'].iloc[station - 1]

    df_reference_stations_u_l = pd.DataFrame()
    df_reference_stations_u_r = pd.DataFrame()
    df_reference_stations_d_l = pd.DataFrame()
    df_reference_stations_d_r = pd.DataFrame()

    list_index = []
    list_distance_nearest = []

# für jeden Quadranten die jeweils nächste Station finden
    for k in range(4):
    
        list_reference_stations_lon = []
        list_reference_stations_lat = []
        list_station = []
        list_distance = []
        
        for i in range(len(coordinates)):
            lon = coordinates['lon'].iloc[i]
            lat = coordinates['lat'].iloc[i]

            # Quadranten definieren
            if k == 0:
                quadrant = (lon >= (lon_station - max_distance_of_reference_stations_lon) and lon <= lon_station) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= lat_station)
            elif k == 1:
                quadrant = (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= lon_station) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= lat_station)
            elif k == 2:
                quadrant = (lon >= (lon_station - max_distance_of_reference_stations_lon) and lon <= lon_station) and (lat >= (lat_station - max_distance_of_reference_stations_lat) and lat <= lat_station)
            elif k == 3:
                quadrant = (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= lon_station) and (lat >= (lat_station - max_distance_of_reference_stations_lat) and lat <= lat_station)
            
            if quadrant:
                if lon == lon_station and lat == lat_station:
                    pass
                else:
                    # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                    # print('\n')

                    distance = round(np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2), 2)

                    list_distance.append(distance)
                    list_station.append(str(i + 1)) #'ams' + str(i)
                    list_reference_stations_lon.append(round(lon, 1))
                    list_reference_stations_lat.append(round(lat, 1))
            
            array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
            df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

        # die nächstgelegene Station herausfiltern
            
        if len(df_reference_stations) == 0:
            # print('No reference stations found')
            pass
        elif len(df_reference_stations) == 1:
            pass
        elif len(df_reference_stations) > 1:
            
            list_distance_nearest_reference_stations = []

            # für jeden Quadranten die jeweils nächste Station finden
            for i in range(len(list_distance)):
                if len(list_distance_nearest_reference_stations) < 1:
                    index_min_distance = np.argmin(list_distance)
                    list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                    del list_distance[index_min_distance]

                    
            # print(list_distance_nearest_reference_stations)

            # df anpassen, sodass nur die nächstgelegene Station übrig bleibt
            for i in df_reference_stations['distance']:
                count = 0
                for j in list_distance_nearest_reference_stations:
                    if i == j:
                        count += 1
                        break
                    else:
                        continue
                if count == 0:
                    df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)
            
            if k == 0:
                df_reference_stations_u_l = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 1:
                df_reference_stations_u_r = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 2:
                df_reference_stations_d_l = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 3:
                df_reference_stations_d_r = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
                
            
            # print(df_reference_stations)
        df_reference_stations.index.name = 'ams'
        df_reference_stations = pd.concat([df_reference_stations_u_l, df_reference_stations_u_r, df_reference_stations_d_l, df_reference_stations_d_r])        
            
    if len(list_index) == 0:
        print('No reference stations found')
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, '-', '-', '-', '-')
    elif len(list_index) == 1:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, int(list_index[0]), '-', '-', '-')
    elif len(list_index) == 2:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, int(list_index[0]), int(list_index[1]), '-', '-')
    elif len(list_index) == 3:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, int(list_index[0]), int(list_index[1]), int(list_index[2]), '-')
    elif len(list_index) == 4:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, int(list_index[0]), int(list_index[1]), int(list_index[2]), int(list_index[3]))

    # check if primary stations are in the range of the secondary station
    
    distance_lon = 10000 # 6400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS
    distance_lat = 10000 # 5400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS

    for i in range(len(loc_prim)):
        lon_check = False
        lat_check = False

        if (loc_prim['lon'][i] < (loc_sec['lon'][station] + distance_lon)) and (loc_prim['lon'][i] > (loc_sec['lon'][station] - distance_lon)):
            lon_check = True
        if (loc_prim['lat'][i] < (loc_sec['lat'][station] + distance_lat)) and (loc_prim['lat'][i] > (loc_sec['lat'][station] - distance_lat)):
            lat_check = True
        if lon_check and lat_check:
            # print('lon:', coordinates_primary_utm32['lon'][i])
            # print('lat:', coordinates_primary_utm32['lat'][i])
            print('primary station', i, 'in range of secondary station', station)
            print('distance to secondary station:', round(np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station])**2), 2))

    return df_reference_stations, list_index, list_distance_nearest

def find_4_nearest_reference_stations_with_plot_and_primary_station(loc_prim, loc_sec, station):
    coordinates = loc_sec
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    # set frame for search
    max_distance_of_reference_stations_lon = 10000
    max_distance_of_reference_stations_lat = 10000

    # set coordinates of the station
    lon_station = coordinates['lon'].iloc[station - 1]
    lat_station = coordinates['lat'].iloc[station - 1]

    list_reference_stations_lon = []
    list_reference_stations_lat = []
    list_station = []
    list_distance = []

    # find the 4 nearest stations in frame
    for i in range(len(coordinates)):
        lon = coordinates['lon'].iloc[i]
        lat = coordinates['lat'].iloc[i]
        if (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= (lon_station - max_distance_of_reference_stations_lon)) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= (lat_station - max_distance_of_reference_stations_lat)):
            if lon == lon_station and lat == lat_station:
                pass
            else:
                # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                # print('\n')

                distance = round(np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2), 2)

                list_distance.append(distance)
                list_station.append(i + 1)
                list_reference_stations_lon.append(round(lon, 2))
                list_reference_stations_lat.append(round(lat, 2))
        
        array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
        df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

# die 4 nächstgelegenen Stationen herausfiltern
        
    if len(df_reference_stations) == 0:
        print('No reference stations found')
        pass
    elif len(df_reference_stations) <= 4:
        for i in range(len(df_reference_stations)):    
            # print(distance)
            pass
    elif len(df_reference_stations) > 4:
        
        list_distance_nearest_reference_stations = []

        for i in range(len(list_distance)):
            if len(list_distance_nearest_reference_stations) < 4:
                index_min_distance = np.argmin(list_distance)
                list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                del list_distance[index_min_distance]

                
        # print(list_distance_nearest_reference_stations)

        for i in df_reference_stations['distance']:
            count = 0
            for j in list_distance_nearest_reference_stations:
                if i == j:
                    count += 1
                    break
                else:
                    continue
            if count == 0:
                df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)
        
    fct.coordinates(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], df_reference_stations.index[1], df_reference_stations.index[2], df_reference_stations.index[3])

    # check if primary stations are in the range of the secondary station
    
    distance_lon = 10000 # 6400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS
    distance_lat = 10000 # 5400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS

    for i in range(len(loc_prim)):
        lon_check = False
        lat_check = False

        if (loc_prim['lon'][i] < (loc_sec['lon'][station] + distance_lon)) and (loc_prim['lon'][i] > (loc_sec['lon'][station] - distance_lon)):
            lon_check = True
        if (loc_prim['lat'][i] < (loc_sec['lat'][station] + distance_lat)) and (loc_prim['lat'][i] > (loc_sec['lat'][station] - distance_lat)):
            lat_check = True
        if lon_check and lat_check:
            # print('lon:', coordinates_primary_utm32['lon'][i])
            # print('lat:', coordinates_primary_utm32['lat'][i])
            print('primary station', i, 'in range of secondary station', station)
            print('distance to secondary station:', round(np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station])**2), 2))


    return df_reference_stations #, list_distance_nearest_reference_stations

def berechnung_Gewichte_für_4_nearest(list_distance_nearest):
    
    list_weights = []

    distance1 = list_distance_nearest.iloc[0]
    distance2 = list_distance_nearest.iloc[1]
    distance3 = list_distance_nearest.iloc[2]
    distance4 = list_distance_nearest.iloc[3]

    W1 = (1/distance1**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
    W2 = (1/distance2**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
    W3 = (1/distance3**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
    W4 = (1/distance4**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
    list_weights.append(W1)
    list_weights.append(W2)
    list_weights.append(W3)
    list_weights.append(W4)

    return list_weights

def berechnung_Referenzniederschlag_4_nearest(secondary_data_df, secondary_data_df_nonan, df_reference_values, list_weights, list_index):
    list_counts_start, list_counts_end, list_counts, list_index_peak = fct.get_data_nan_seq_before_peak(secondary_data_df, 'sc', 'ams6', 0.99)

    for c, p in zip(range(0, len(list_counts_start)), range(0, len(list_index_peak))):    
        for index in secondary_data_df_nonan.loc[list_counts_start[c] : list_index_peak[p]].index:
            h1 = secondary_data_df_nonan['ams' + str(list_index[0])].loc[index]
            h2 = secondary_data_df_nonan['ams' + str(list_index[1])].loc[index]
            h3 = secondary_data_df_nonan['ams' + str(list_index[2])].loc[index]
            h4 = secondary_data_df_nonan['ams' + str(list_index[3])].loc[index]

            h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3 + list_weights[3]*h4

            df_reference_values.loc[index] = h_ref

    return df_reference_values

def calculate_correlation_with_without_resample(data_primary, data_secondary, prim_station, sec_station, percentile, resample_sec, resample_prim):
    if resample_sec == True:
        df_agg = fct.resampleDf(data_secondary[sec_station], 'H', closed='right', label='right', shift=False, leave_nan=True, label_shift=None, temp_shift=0, max_nan=2)
    else:
        df_agg = data_secondary[sec_station]

    index_start = data_secondary.index[0]
    index_end = data_secondary.index[-1]

    if resample_prim == True:
        df_reference = fct.resampleDf(data_primary[prim_station][index_start:index_end], 'H', closed='right', label='right', shift=False, leave_nan=True, label_shift=None, temp_shift=0, max_nan=2)
    else:
        df_reference = data_primary[prim_station][index_start:index_end]

    df_for_correlation = pd.concat([df_reference, df_agg], axis=1)
    df_for_correlation = df_for_correlation.dropna() # ohne wäre correlation nan

    cc, a_dataset, b_dataset = fct.calc_indicator_correlation(df_for_correlation.iloc[:, 0].values, df_for_correlation.iloc[:, 1].values, percentile)

    return cc

def correction_complete_amsterdam(sec_utm, station_zahl, secondary_data_df_nonan, secondary_data_df):

    '''
    0.99 Perzentil bei find nan sequence before peak
    5 km max Entfernung zu den Referenzstationen
    '''

    reference_stations = fct.find_4_nearest_reference_stations(sec_utm, station_zahl)
    list_weights = fct.berechnung_Gewichte_für_4_nearest(list_distance_nearest=reference_stations['distance'])
    df_reference_values = secondary_data_df[['ams' + str(station_zahl)]].copy()
    df_reference_values_calculated = fct.berechnung_Referenzniederschlag_4_nearest(secondary_data_df, secondary_data_df_nonan, df_reference_values, list_weights, reference_stations.index)
    data_corrected = fct.correct_data(secondary_data_df, df_reference_values_calculated, 'sc', 'ams' + str(station_zahl), 0.99)

    return data_corrected

def coordinates_with_find_primary(loc_prim, loc_sec, y, station, ref1, ref2, ref3, ref4, prim_in_sec=False):
    
    if y == 'primary':
        coords_lon = loc_prim['lon']
        coords_lat = loc_prim['lat']
    elif y == 'secondary':
        coords_lon = loc_sec['lon']
        coords_lat = loc_sec['lat']
    elif y == 'both':
        coords_lon_prim = loc_prim['lon']
        coords_lat_prim = loc_prim['lat']
        coords_lon_sec = loc_sec['lon']
        coords_lat_sec = loc_sec['lat']

    if y == 'both':
        name_plot = 'Coordinates ' + y + ' networks'
        plt.scatter(x=coords_lon_prim, y=coords_lat_prim, s=20, color='red', label='primary network', marker='x', linewidth=1)
        plt.scatter(x=coords_lon_sec, y=coords_lat_sec, s=2, color='blue', label='secondary network', alpha=0.5)
        if type(station) == int:
            plt.scatter(loc_prim['lon'].iloc[station], loc_prim['lat'].iloc[station], color='black')
        plt.legend()
    else:
        name_plot = 'Coordinates ' + y + ' network'
        plt.scatter(x=coords_lon, y=coords_lat, s=10)
        if type(station) == int:
            if y == 'primary':
                plt.scatter(loc_prim['lon'].iloc[station], loc_prim['lat'].iloc[station], color='red')
            elif y == 'secondary':
                plt.scatter(loc_sec['lon'].iloc[station - 1], loc_sec['lat'].iloc[station - 1], color='red')

                if prim_in_sec == True:
                    # check if primary stations are in the range of the secondary station
                        distance_lon = 10000 # 6400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS
                        distance_lat = 10000 # 5400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS

                        for i in range(len(loc_prim)):
                            lon_check = False
                            lat_check = False

                            if (loc_prim['lon'][i] < (loc_sec['lon'][station] + distance_lon)) and (loc_prim['lon'][i] > (loc_sec['lon'][station] - distance_lon)):
                                lon_check = True
                            if (loc_prim['lat'][i] < (loc_sec['lat'][station] + distance_lat)) and (loc_prim['lat'][i] > (loc_sec['lat'][station] - distance_lat)):
                                lat_check = True
                            if lon_check and lat_check:
                                # print('lon:', coordinates_primary_utm32['lon'][i])
                                # print('lat:', coordinates_primary_utm32['lat'][i])
                                print('primary station', i, 'in range of secondary station', station)
                                print('distance to secondary station:', round(np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station])**2), 2))

                                plt.scatter(loc_prim['lon'].iloc[i], loc_prim['lat'].iloc[i], color='black', marker='x', linewidth=1)

            try:
                plt.scatter(loc_sec['lon'].iloc[ref1 - 1], loc_sec['lat'].iloc[ref1 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref2 - 1], loc_sec['lat'].iloc[ref2 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref3 - 1], loc_sec['lat'].iloc[ref3 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref4 - 1], loc_sec['lat'].iloc[ref4 - 1], color='lime', s=10)
            except:
                pass

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(name_plot)

    plt.show()
    plt.close()
    
    return # print(coords_lon, coords_lat)

def correction_complete_amsterdam_with_primary(sec_utm, prim_utm, station_zahl, secondary_data_df_nonan, secondary_data_df, primary_data_df):
       
    # check if primary stations are in the range of the secondary station
    
    distance_lon = 5000 # 6400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS
    distance_lat = 5000 # 5400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS

    count = 0

    for i in range(len(prim_utm)):
        lon_check = False
        lat_check = False

        if (prim_utm['lon'][i] < (sec_utm['lon'][station_zahl] + distance_lon)) and (prim_utm['lon'][i] > (sec_utm['lon'][station_zahl] - distance_lon)):
            lon_check = True
        if (prim_utm['lat'][i] < (sec_utm['lat'][station_zahl] + distance_lat)) and (prim_utm['lat'][i] > (sec_utm['lat'][station_zahl] - distance_lat)):
            lat_check = True
        if lon_check and lat_check:
            df_reference_values = primary_data_df[[i]].copy().rename(columns={i : 'ams' + str(station_zahl)})
            count += 1

    if count == 0:
        reference_stations = fct.find_4_nearest_reference_stations(sec_utm, station_zahl)
        list_weights = fct.berechnung_Gewichte_für_4_nearest(list_distance_nearest=reference_stations['distance'])
        df_reference_values = secondary_data_df[['ams' + str(station_zahl)]].copy()
        df_reference_values_calculated = fct.berechnung_Referenzniederschlag_4_nearest(secondary_data_df, secondary_data_df_nonan, df_reference_values, list_weights, reference_stations.index)
        data_corrected = fct.correct_data(secondary_data_df, df_reference_values_calculated, 'sc', 'ams' + str(station_zahl), 0.99)

    if count != 0:
        secondary_data_df_res = fct.resampleDf(secondary_data_df[['ams' + str(station_zahl)]], 'H', closed='right', label='right', shift=False, leave_nan=True, label_shift=None, temp_shift=0, max_nan=2)
        
        index_start = secondary_data_df[['ams' + str(station_zahl)]].index[0]
        index_end = secondary_data_df[['ams' + str(station_zahl)]].index[-1]

        df_reference_values_correct_index = df_reference_values[index_start:index_end]

        data_corrected = fct.correct_data(secondary_data_df_res, df_reference_values_correct_index, 'pr', 'ams' + str(station_zahl), 0.99)   

    return data_corrected

def list_nan_sequences(dataframe, column, min_len):

    dataframe_mask = dataframe.isna()
    column = column

    list = []

    count = 0
    index_count = 0

    for value in dataframe_mask[column]:
        if value == True:
            if count == 0:    
                index_start = dataframe[column].index[index_count]
                count += 1
            else:
                count += 1
                if dataframe[column].index[index_count] == dataframe[column].index[-1]:
                    index_end = dataframe[column].index[index_count]
                    if count < min_len:
                        pass
                    else:
                        list.append(count) # , index_start, index_end])

        else:
            if count == 0:
                pass
            else:
                index_end = dataframe[column].index[index_count - 1]
                if count < min_len:
                    pass
                else:
                    list.append(count) # , index_start, index_end])
                count = 0
        index_count += 1

    # print('Count of nan sequences with min len ' + str(min_len) + ':', len(list))

    return list

def histogramm_balken(data, column):

    station = fct.list_nan_sequences(data, column, 1)

    fig, ax = plt.subplots()

    bins = [1, 10, 100, 1000, 10000, 100000]
    hist = np.histogram(station, bins=bins)[0]

    ax.hist(station, bins=bins, linewidth=0.5, edgecolor="white")

    plt.xscale('log')
    ticks = [10**i for i in range(6)]  # Erzeugt eine Liste [1, 10, 100, 1000]
    plt.xticks(ticks, labels=ticks)  # Setzt die x-Achsenwerte und -Beschriftungen

    plt.yscale('log')
    ticks = [10**i for i in range(6)]  # Erzeugt eine Liste [1, 10, 100, 1000]
    plt.yticks(ticks, labels=ticks)  # Setzt die x-Achsenwerte und -Beschriftungen

    plt.xlabel('Länge der nan-Sequenz')
    plt.ylabel('Vorkommen der Länge')
    plt.title('Vorkommen der Längen der nan-Sequenzen in 10*i Intervallen')

    plt.show()

    return

def histogramm_scatter(data, column):

    station = fct.list_nan_sequences(data, column, 1)

    absolute_frequencies = np.unique(station, return_counts=True)

    x = absolute_frequencies[0]
    y = absolute_frequencies[1]

    hist = np.histogram(station, bins=[10**i for i in range(6)])[0]

    # plot
    fig, ax = plt.subplots()

    ax.scatter(x, y, marker='.', alpha=0.5, edgecolors='none', s=100)

    plt.hlines(hist[0], 1, 10, color='r', linestyle='--', label='Vorkommen im Intervall 10**i')
    plt.hlines(hist[1], 10, 100, color='r', linestyle='--')
    plt.hlines(hist[2], 100, 1000, color='r', linestyle='--')
    plt.hlines(hist[3], 1000, 10000, color='r', linestyle='--')
    plt.hlines(hist[4], 10000, 100000, color='r', linestyle='--')

    plt.vlines(1, 0, hist[0], color='r', linestyle='--')
    plt.vlines(10, 0, hist[0], color='r', linestyle='--')
    plt.vlines(10, 0, hist[1], color='r', linestyle='--')
    plt.vlines(100, 0, hist[1], color='r', linestyle='--')
    plt.vlines(100, 0, hist[2], color='r', linestyle='--')
    plt.vlines(1000, 0, hist[2], color='r', linestyle='--')
    plt.vlines(1000, 0, hist[3], color='r', linestyle='--')
    plt.vlines(10000, 0, hist[3], color='r', linestyle='--')
    plt.vlines(10000, 0, hist[4], color='r', linestyle='--')
    plt.vlines(100000, 0, hist[4], color='r', linestyle='--')

    plt.xscale('log')
    ticks = [10**i for i in range(6)]  # Erzeugt eine Liste [1, 10, 100, 1000]
    plt.xticks(ticks, labels=ticks)  # Setzt die x-Achsenwerte und -Beschriftungen

    plt.yscale('log')
    plt.yticks(ticks, labels=ticks)  # Setzt die y-Achsenwerte und -Beschriftungen

    plt.xlabel('Länge der nan-Sequenz')
    plt.ylabel('Vorkommen der Länge')
    plt.title('Vorkommen der Länge der nan-Sequenzen')

    plt.legend()

    plt.show()

    return

def einer_zweier_sequ_korrigieren(data, column, _1seq=True, _2seq=True):
    # nan die alleine Stehen oder nan-Paare zu 0 machen

    station_touse = data

    station = station_touse[[column]].copy()

    # True für alleinstehende nans
    mask_1seq = station.isna() & station.shift(-1).notnull() & station.shift(1).notnull()

    # erste und letzte Werte extra betrachten
    if (station.iloc[0].isna() & station.iloc[1].notnull()).bool():
        mask_1seq.iloc[0] = True
    if (station.iloc[-1].isna() & station.iloc[-2].notnull()).bool():
        mask_1seq.iloc[-1] = True

    # True für nan-Paare
    mask1 = station.isna() & station.shift(-1).notnull() & station.shift(1).isna() & station.shift(2).notnull()
    mask2 = station.isna() & station.shift(-1).isna() & station.shift(1).notnull() & station.shift(-2).notnull()
    maske_2seq = mask1 | mask2

    # erste zwei und letzte zwei Werte extra betrachten
    if (station.iloc[0].isna() & station.iloc[1].isna() & station.iloc[2].notnull()).bool():
        maske_2seq.iloc[0] = True
        maske_2seq.iloc[1] = True
    if (station.iloc[-1].isna() & station.iloc[-2].isna() & station.iloc[-3].notnull()).bool():
        maske_2seq.iloc[-1] = True
        maske_2seq.iloc[-2] = True

    if _1seq == True:
        station[mask_1seq] = 0
    if _2seq == True:
        station[maske_2seq] = 0
    
    return station

def list_nan_sequences_schnell(data, station, timedelta):
    
    '''starts, ends, len_seq'''
    
    if timedelta == '1min':
        timedelta = datetime.timedelta(minutes=1)
    if timedelta == '5min':
        timedelta = datetime.timedelta(minutes=5)
    if timedelta == '1h':
        timedelta = datetime.timedelta(hours=1)

    is_nan = data[station].isna() # gibt true zurück, wenn Wert NaN ist
    diff = is_nan.diff() # gibt true zurück, wenn Wert zu Nan oder Nan zu Wert springt

    # print(is_nan)
    # print(diff)

    if is_nan[0] == True:
        diff[0] = True

    starts = diff[diff == True].index[::2]
    ends = diff[diff == True].index[1::2] - timedelta

    if is_nan[-1] == True:
        ends = ends.append(data.index[-1:])
    elif len(starts) > len(ends):
        starts = starts.delete(-1)

    len_seq = ((ends + timedelta) - starts)/timedelta
    len_seq = len_seq.astype(int)
    
    # print('starts:', starts)
    # print('ends:', ends)
    
    return starts, ends, len_seq


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# Funktionen aus Bachelor funktions

def sum_curve(data, station, date_start, date_end):

    # set DataFrame for this run of this funktion
    dataframe = data[station][date_start:date_end]
    
    # Messwerte der Station aufsummieren
    sum_list = dataframe.cumsum().tolist()
    sum_total = dataframe.sum()

    # "Summendataframe" erstellen
    index_sum_df = dataframe.index
    sum_df = pd.Series(data=sum_list, index=index_sum_df)

    # Subplot erstellen
    fig, ax = plt.subplots()

    # plot Tageswerte
    ax.plot(index_sum_df, dataframe, label='Tageswerte', color='green')
    ax.set_ylabel('Tageswert [mm]')
    plt.xticks(rotation=45)
    plt.legend(loc=9)

    # plot Summenkurve
    ax = ax.twinx()
    ax.plot(index_sum_df, sum_list, label='Summenkurve', color='red')
    ax.set_ylabel('Summenkurve [mm]')
    plt.legend(loc=2)

    name_plot = 'Summenkurve + Tageswerte, Station: ' + str(station)
    plt.title(name_plot)
    plt.xlabel('DateTime')
    plt.legend()
    
    plt.show()
    plt.close()
    
    return print('Gesamtniederschlag über Zeitraum:', round(sum_total, 2), 'mm\n') # (print('Index Station', str(station), ': \n\n', index_sum_df, '\n\n', 'Summe aktuel zu Zeitstempel:\n\n', sum_df))

def nan_nonan_ratio(data, station):

    dataframe = data[station]
    
    nan_count = dataframe.isna().sum()
    ratio = (nan_count / len(dataframe))*100
    print('count of nans:', nan_count, '\nlength of data:', len(dataframe), '\nRatio of nan/nonan:', round(ratio, 2), '%')

    return

def longest_nan_sequence(data, station):

    dataframe = data[station].isna()
    
    longest_sequences = {}
    
    longest_sequence_length = 0
    current_sequence_length = 0
      
    for value in dataframe:
        if value == True:  # Wenn der Wert NaN ist
            current_sequence_length += 1
            if current_sequence_length > longest_sequence_length:
                longest_sequence_length = current_sequence_length
        else:
            current_sequence_length = 0
        
    longest_sequences[station] = longest_sequence_length
    
    return longest_sequences

def find_nan_sequence(data, station, day_start, day_end, max_nans):
    # 17*nan = 1h

    dataframe = data[station][day_start : day_end]
    position_in_dataframe = 0
    count = 0
    
    for value in dataframe:
        if np.isnan(value) == True:
            if count == 0:
                interval_start = dataframe.index[position_in_dataframe]
                
            count += 1
            if count == max_nans:
                interval_end = dataframe.index[position_in_dataframe]
                print('nan sequence of', max_nans, 'nans', '\ntime interval:', interval_start, ' to', interval_end)
                count = 0
            position_in_dataframe += 1
        else:
            position_in_dataframe += 1
            count = 0
            continue
            
    return

def i_nans_before_peak(data, y, station, quantile):

    if y == 'pr':
        timegap = datetime.timedelta(hours=1)
    elif y == 'sc':
        timegap = datetime.timedelta(minutes=5)

    dataframe = data[station]

    peaks = dataframe[dataframe > dataframe.quantile(quantile)]
    
    for index_peak in peaks.index:
        count = 0
        for i in reversed(dataframe.loc[: index_peak - timegap].isna()):
            if i == True:
                count += 1
            else:
                if count > 0:
                    print(count, 'leading nans before', index_peak)
                    break
                else:
                    # print('no leading nans before', index_peak)
                    break      
    return

def coordinates(loc_prim, loc_sec, y, station, ref1, ref2, ref3, ref4):
    
    if y == 'primary':
        coords_lon = loc_prim['lon']
        coords_lat = loc_prim['lat']
    elif y == 'secondary':
        coords_lon = loc_sec['lon']
        coords_lat = loc_sec['lat']
    elif y == 'both':
        coords_lon_prim = loc_prim['lon']
        coords_lat_prim = loc_prim['lat']
        coords_lon_sec = loc_sec['lon']
        coords_lat_sec = loc_sec['lat']

    if y == 'both':
        name_plot = 'Coordinates ' + y + ' networks'
        plt.scatter(x=coords_lon_prim, y=coords_lat_prim, s=20, color='red', label='primary network', marker='x', linewidth=1)
        plt.scatter(x=coords_lon_sec, y=coords_lat_sec, s=2, color='blue', label='secondary network', alpha=0.5)
        if type(station) == int:
            plt.scatter(loc_prim['lon'].iloc[station], loc_prim['lat'].iloc[station], color='black')
        plt.legend()
    else:
        name_plot = 'Coordinates ' + y + ' network'
        plt.scatter(x=coords_lon, y=coords_lat, s=10)
        if type(station) == int:
            if y == 'primary':
                plt.scatter(loc_prim['lon'].iloc[station], loc_prim['lat'].iloc[station], color='red')
            elif y == 'secondary':
                plt.scatter(loc_sec['lon'].iloc[station - 1], loc_sec['lat'].iloc[station - 1], color='red')

            try:
                plt.scatter(loc_sec['lon'].iloc[ref1 - 1], loc_sec['lat'].iloc[ref1 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref2 - 1], loc_sec['lat'].iloc[ref2 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref3 - 1], loc_sec['lat'].iloc[ref3 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref4 - 1], loc_sec['lat'].iloc[ref4 - 1], color='lime', s=10)
            except:
                pass

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(name_plot)

    plt.show()
    plt.close()
    
    return # print(coords_lon, coords_lat)

def find_primary_stations(loc_prim, lon_x1, lon_x2, lat_y1, lat_y2):
    
    for i in range(len(loc_prim)):
        lon = loc_prim['lon'][i]
        lat = loc_prim['lat'][i]
        if (lon <= lon_x2 and lon >= lon_x1) and (lat <= lat_y2 and lat >= lat_y1):
            print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
    
    return

def get_data_nan_seq_before_peak(data, y, station, quantile):

    '''
    outside of function, write:
    output_list_counts_start, output_list_counts_end, output_list_counts, output_list_index_peak = get_data_nan_seq_before_peak(...)
    '''

    if y == 'pr':
        timegap = datetime.timedelta(hours=1)
    elif y == 'sc':
        timegap = datetime.timedelta(minutes=5)

    dataframe = data[station]

    peaks = dataframe[dataframe > dataframe.quantile(quantile)]
    list_index_peak = []
    list_counts = []
    list_counts_start = []
    list_counts_end = []

    for index_peak in peaks.index:
        count = 0
        for i in reversed(dataframe.loc[: index_peak - timegap].isna()):
            if i == True:
                count += 1
            else:
                if count > 0:
                    count_start = index_peak - (timegap * count)
                    list_counts_start.append(count_start)
                    list_counts_end.append(index_peak - timegap)
                    list_counts.append(count)
                    list_index_peak.append(index_peak)
                    break
                else:
                    break      
    return list_counts_start, list_counts_end, list_counts, list_index_peak

def correct_data(data, reference, y, station, quantile):

    # fct.get_data_nan_seq_before_peak(data, y, station, quantile) # to get the output lists
    output_list_counts_start, output_list_counts_end, output_list_counts, output_list_index_peak = fct.get_data_nan_seq_before_peak(data, y, station, quantile)
   
    data_corrected = data[[station]].copy() # copy the data to a new dataframe

    if y == 'pr':
        frequency = '1h'
    elif y == 'sc':
        frequency = '5min'

    for i in range(len(output_list_index_peak)):

        datetime_index = pd.date_range(start=output_list_counts_start[i], end=output_list_index_peak[i], freq=frequency) # create a datetime index for the time period of the nan sequence before the peak
        sum = reference[station].loc[output_list_counts_start[i] : output_list_index_peak[i]].sum() # sum of the reference values for the time period of the nan sequence before the peak
        value_peak = data[station].loc[output_list_index_peak[i]] # value of the peak

        for index in datetime_index:
            try:
                peak_portion = round(((reference[station].loc[index] / sum) * value_peak), 2)
            except ZeroDivisionError:
                peak_portion = 0
                
            data_corrected[station].loc[index] = peak_portion # replace the nan values with the calculated peak portion
        
    return data_corrected

def find_4_nearest_reference_stations_new(coordinates, station):
    
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    # set frame for search
    radius = 1500

    # set coordinates of the station
    lon_station = coordinates['lon'].iloc[station - 1]
    lat_station = coordinates['lat'].iloc[station - 1]

    list_reference_stations_lon = []
    list_reference_stations_lat = []
    list_station = []
    list_distance = []

    # find the 4 nearest stations in frame
    for i in range(len(coordinates)):
        lon = coordinates['lon'].iloc[i]
        lat = coordinates['lat'].iloc[i]
        if (np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2) <= radius):
            if lon == lon_station and lat == lat_station:
                pass
            else:
                # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                # print('\n')

                distance = np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2)

                list_distance.append(distance)
                list_station.append(i + 1)
                list_reference_stations_lon.append(lon)
                list_reference_stations_lat.append(lat)
        
        array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
        df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

# die 4 nächstgelegenen Stationen herausfiltern
        
    if len(df_reference_stations) == 0:
        print('No reference stations found')
        pass
    elif len(df_reference_stations) <= 4:
        for i in range(len(df_reference_stations)):    
            # print(distance)
            pass
    elif len(df_reference_stations) > 4:
        
        list_distance_nearest_reference_stations = []

        for i in range(len(list_distance)):
            if len(list_distance_nearest_reference_stations) < 4:
                index_min_distance = np.argmin(list_distance)
                list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                del list_distance[index_min_distance]

                
        # print(list_distance_nearest_reference_stations)

        for i in df_reference_stations['distance']:
            count = 0
            for j in list_distance_nearest_reference_stations:
                if i == j:
                    count += 1
                    break
                else:
                    continue
            if count == 0:
                df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)

    return df_reference_stations

def find_reference_stations_for_each_quadrant(coordinates, station):
    
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    max_distance_of_reference_stations_lon = 10000000000
    max_distance_of_reference_stations_lat = 10000000000

    lon_station = coordinates['lon'].iloc[station - 1]
    lat_station = coordinates['lat'].iloc[station - 1]

    df_reference_stations_u_l = pd.DataFrame()
    df_reference_stations_u_r = pd.DataFrame()
    df_reference_stations_d_l = pd.DataFrame()
    df_reference_stations_d_r = pd.DataFrame()

    list_index = []
    list_distance_nearest = []

# für jeden Quadranten die jeweils nächste Station finden
    for k in range(4):
    
        list_reference_stations_lon = []
        list_reference_stations_lat = []
        list_station = []
        list_distance = []
        
        for i in range(len(coordinates)):
            lon = coordinates['lon'].iloc[i]
            lat = coordinates['lat'].iloc[i]

            # Quadranten definieren
            if k == 0:
                quadrant = (lon >= (lon_station - max_distance_of_reference_stations_lon) and lon <= lon_station) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= lat_station)
            elif k == 1:
                quadrant = (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= lon_station) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= lat_station)
            elif k == 2:
                quadrant = (lon >= (lon_station - max_distance_of_reference_stations_lon) and lon <= lon_station) and (lat >= (lat_station - max_distance_of_reference_stations_lat) and lat <= lat_station)
            elif k == 3:
                quadrant = (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= lon_station) and (lat >= (lat_station - max_distance_of_reference_stations_lat) and lat <= lat_station)
            
            if quadrant:
                if lon == lon_station and lat == lat_station:
                    pass
                else:
                    # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                    # print('\n')

                    distance = np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2)

                    list_distance.append(distance)
                    list_station.append('ams' + str(i + 1))
                    list_reference_stations_lon.append(lon)
                    list_reference_stations_lat.append(lat)
            
            array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
            df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

        # die nächstgelegene Station herausfiltern
            
        if len(df_reference_stations) == 0:
            # print('No reference stations found')
            pass
        elif len(df_reference_stations) == 1:
            pass
        elif len(df_reference_stations) > 1:
            
            list_distance_nearest_reference_stations = []

            # für jeden Quadranten die jeweils nächste Station finden
            for i in range(len(list_distance)):
                if len(list_distance_nearest_reference_stations) < 1:
                    index_min_distance = np.argmin(list_distance)
                    list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                    del list_distance[index_min_distance]

                    
            # print(list_distance_nearest_reference_stations)

            # df anpassen, sodass nur die nächstgelegene Station übrig bleibt
            for i in df_reference_stations['distance']:
                count = 0
                for j in list_distance_nearest_reference_stations:
                    if i == j:
                        count += 1
                        break
                    else:
                        continue
                if count == 0:
                    df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)
            
            if k == 0:
                df_reference_stations_u_l = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 1:
                df_reference_stations_u_r = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 2:
                df_reference_stations_d_l = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 3:
                df_reference_stations_d_r = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
                
            
            # print(df_reference_stations)
        df_reference_stations = pd.concat([df_reference_stations_u_l, df_reference_stations_u_r, df_reference_stations_d_l, df_reference_stations_d_r])        
            
    return list_index, list_distance_nearest, df_reference_stations

def berechnung_Gewichte(list_index, list_distance_nearest):
    
    list_weights = []

    if len(list_index) == 0:
        W1 = 0
        W2 = 0
        W3 = 0
        W4 = 0
    elif len(list_index) == 1:
        distance1 = list_distance_nearest.iloc[0]

        W1 = 1
        list_weights.append(W1)
    elif len(list_index) == 2:
        distance1 = list_distance_nearest.iloc[0]
        distance2 = list_distance_nearest.iloc[1]

        W1 = (1/distance1**2)/((1/distance1**2)+(1/distance2**2))
        W2 = (1/distance2**2)/((1/distance1**2)+(1/distance2**2))
        list_weights.append(W1)
        list_weights.append(W2)
    elif len(list_index) == 3:
        distance1 = list_distance_nearest.iloc[0]
        distance2 = list_distance_nearest.iloc[1]
        distance3 = list_distance_nearest.iloc[2]

        W1 = (1/distance1**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2))
        W2 = (1/distance2**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2))
        W3 = (1/distance3**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2))
        list_weights.append(W1)
        list_weights.append(W2)
        list_weights.append(W3)
    elif len(list_index) == 4:
        distance1 = list_distance_nearest.iloc[0]
        distance2 = list_distance_nearest.iloc[1]
        distance3 = list_distance_nearest.iloc[2]
        distance4 = list_distance_nearest.iloc[3]

        W1 = (1/distance1**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
        W2 = (1/distance2**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
        W3 = (1/distance3**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
        W4 = (1/distance4**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
        list_weights.append(W1)
        list_weights.append(W2)
        list_weights.append(W3)
        list_weights.append(W4)

    return list_weights

def berechnung_Referenzniederschlag(secondary_data_df_nonan, df_reference_values, list_weights, list_index):
    
    if list_weights == []:
        pass
    elif len(list_weights) == 1:
        for index in secondary_data_df_nonan.index:
            h1 = secondary_data_df_nonan[list_index[0]].loc[index]

            h_ref = list_weights[0]*h1

            df_reference_values.loc[index] = h_ref
    elif len(list_weights) == 2:
        for index in secondary_data_df_nonan.index:
            h1 = secondary_data_df_nonan[list_index[0]].loc[index]
            h2 = secondary_data_df_nonan[list_index[1]].loc[index]

            h_ref = list_weights[0]*h1 + list_weights[1]*h2

            df_reference_values.loc[index] = h_ref
    elif len(list_weights) == 3:
        for index in secondary_data_df_nonan.index:
            h1 = secondary_data_df_nonan[list_index[0]].loc[index]
            h2 = secondary_data_df_nonan[list_index[1]].loc[index]
            h3 = secondary_data_df_nonan[list_index[2]].loc[index]

            h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3

            df_reference_values.loc[index] = h_ref
    elif len(list_weights) == 4:
        for index in secondary_data_df_nonan.index:
            h1 = secondary_data_df_nonan[list_index[0]].loc[index]
            h2 = secondary_data_df_nonan[list_index[1]].loc[index]
            h3 = secondary_data_df_nonan[list_index[2]].loc[index]
            h4 = secondary_data_df_nonan[list_index[3]].loc[index]

            h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3 + list_weights[3]*h4

            df_reference_values.loc[index] = h_ref

    return df_reference_values

def calculate_reference_df(data_prim):
    index = pd.date_range(start='2019-12-31 23:05:00', end='2023-10-14 21:00:00', freq='5min')
    df_reference = pd.DataFrame(index=index, columns=['station'])
    df_reference

    for index in df_reference.index:
    
        try:
            h1 = data_prim['rr07'].loc[index]
            h2 = data_prim['rr10'].loc[index]

            h_ref = 0.5*h1 + 0.5*h2

            df_reference.loc[index] = h_ref
        except KeyError:
            df_reference.loc[index] = np.nan
    return df_reference

def correct_data_lauchaecker(data, df_reference, station, quantile):
        
    df_reference = df_reference.rename(columns={df_reference.columns[0] : station})
    data_corrected = fct.correct_data_new(data, df_reference, 'sc', station, quantile)

    return data_corrected

def LatLon_To_XY(i_area, j_area):
    
    ''' convert coordinates from wgs84 to utm 32'''
    P = pyproj.Proj(proj='utm', zone=32,
                    ellps='WGS84',
                    preserve_units=True)

    x, y = P.transform(i_area, j_area)

    return x, y

def resampleDf(df, agg, closed='right', label='right',
               shift=False, leave_nan=True,
               label_shift=None,
               temp_shift=0,
               max_nan=0):

    if shift == True:
        df_copy = df.copy()
        if agg != 'D' and agg != '1440min':
            raise Exception('Shift can only be applied to daily aggregations')
        df = df.shift(-6, 'H')

    # To respect the nan values
    if leave_nan == True:
        # for max_nan == 0, the code runs faster if implemented as follows
        if max_nan == 0:
            # print('Resampling')
            # Fill the nan values with values very great negative values and later
            # get the out again, if the sum is still negative
            df = df.fillna(-100000000000.)
            df_agg = df.resample(agg,
                                 closed=closed,
                                 label=label,
                                 offset=temp_shift).sum()
            # Replace negative values with nan values
            df_agg.values[df_agg.values[:] < 0.] = np.nan
        else:
            df_agg = df.resample(rule=agg,
                                 closed=closed,
                                 label=label,
                                 offset=temp_shift).sum()
            # find data with nan in original aggregation
            g_agg = df.groupby(pd.Grouper(freq=agg,
                                          closed=closed,
                                          label=label))
            n_nan_agg = g_agg.aggregate(lambda x: pd.isnull(x).sum())

            # set aggregated data to nan if more than max_nan values occur in the
            # data to be aggregated
            filter_nan = (n_nan_agg > max_nan)
            df_agg[filter_nan] = np.nan

    elif leave_nan == False:
        df_agg = df.resample(agg,
                             closed=closed,
                             label=label).sum()
    if shift == True:
        df = df_copy
    return df_agg

def calc_indicator_correlation(a_dataset, b_dataset, prob):
    """
    Tcalcualte indicator correlation two datasets

    Parameters
    ----------
    a_dataset: first data vector
    b_dataset: second data vector
    perc: percentile threshold 
    
    Returns
    ----------
    indicator correlation value

    Raises
    ----------

    """
    a_sort = np.sort(a_dataset)
    b_sort = np.sort(b_dataset)
    ix = int(a_dataset.shape[0] * prob)
    a_dataset[a_dataset < a_sort[ix]] = 0 # a_sort[ix] liefert perzentilwert abhängig von prob
    b_dataset[b_dataset < b_sort[ix]] = 0
    a_dataset[a_dataset > 0] = 1
    b_dataset[b_dataset > 0] = 1
    cc = np.corrcoef(a_dataset, b_dataset)[0, 1]

    return cc, a_dataset, b_dataset

def calculate_correlation_with_resample(data_primary, data_secondary, prim_station, sec_station, percentile):
    df_agg = fct.resampleDf(data_secondary[sec_station], 'H', closed='right', label='right', shift=False, leave_nan=True, label_shift=None, temp_shift=0, max_nan=2)

    index_start = df_agg.index[0]
    index_end = df_agg.index[-1]

    df_reference = data_primary[prim_station][index_start:index_end]

    df_for_correlation = pd.concat([df_reference, df_agg], axis=1)
    df_for_correlation = df_for_correlation.dropna() # ohne wäre correlation nan

    cc, a_dataset, b_dataset = fct.calc_indicator_correlation(df_for_correlation.iloc[:, 0], df_for_correlation.iloc[:, 1], percentile)

    return cc

def find_reference_stations_for_each_quadrant_with_plot_and_primary_station(loc_prim, loc_sec, station):
    
    coordinates = loc_sec
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    max_distance_of_reference_stations_lon = 3000
    max_distance_of_reference_stations_lat = 3000

    lon_station = coordinates['lon'].iloc[station - 1]
    lat_station = coordinates['lat'].iloc[station - 1]

    df_reference_stations_u_l = pd.DataFrame()
    df_reference_stations_u_r = pd.DataFrame()
    df_reference_stations_d_l = pd.DataFrame()
    df_reference_stations_d_r = pd.DataFrame()

    list_index = []
    list_distance_nearest = []

# für jeden Quadranten die jeweils nächste Station finden
    for k in range(4):
    
        list_reference_stations_lon = []
        list_reference_stations_lat = []
        list_station = []
        list_distance = []
        
        for i in range(len(coordinates)):
            lon = coordinates['lon'].iloc[i]
            lat = coordinates['lat'].iloc[i]

            # Quadranten definieren
            if k == 0:
                quadrant = (lon >= (lon_station - max_distance_of_reference_stations_lon) and lon <= lon_station) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= lat_station)
            elif k == 1:
                quadrant = (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= lon_station) and (lat <= (lat_station + max_distance_of_reference_stations_lat) and lat >= lat_station)
            elif k == 2:
                quadrant = (lon >= (lon_station - max_distance_of_reference_stations_lon) and lon <= lon_station) and (lat >= (lat_station - max_distance_of_reference_stations_lat) and lat <= lat_station)
            elif k == 3:
                quadrant = (lon <= (lon_station + max_distance_of_reference_stations_lon) and lon >= lon_station) and (lat >= (lat_station - max_distance_of_reference_stations_lat) and lat <= lat_station)
            
            if quadrant:
                if lon == lon_station and lat == lat_station:
                    pass
                else:
                    # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                    # print('\n')

                    distance = round(np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2), 2)

                    list_distance.append(distance)
                    list_station.append(str(i + 1)) #'ams' + str(i)
                    list_reference_stations_lon.append(round(lon, 1))
                    list_reference_stations_lat.append(round(lat, 1))
            
            array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
            df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

        # die nächstgelegene Station herausfiltern
            
        if len(df_reference_stations) == 0:
            # print('No reference stations found')
            pass
        elif len(df_reference_stations) == 1:
            pass
        elif len(df_reference_stations) > 1:
            
            list_distance_nearest_reference_stations = []

            # für jeden Quadranten die jeweils nächste Station finden
            for i in range(len(list_distance)):
                if len(list_distance_nearest_reference_stations) < 1:
                    index_min_distance = np.argmin(list_distance)
                    list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                    del list_distance[index_min_distance]

                    
            # print(list_distance_nearest_reference_stations)

            # df anpassen, sodass nur die nächstgelegene Station übrig bleibt
            for i in df_reference_stations['distance']:
                count = 0
                for j in list_distance_nearest_reference_stations:
                    if i == j:
                        count += 1
                        break
                    else:
                        continue
                if count == 0:
                    df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)
            
            if k == 0:
                df_reference_stations_u_l = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 1:
                df_reference_stations_u_r = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 2:
                df_reference_stations_d_l = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
            elif k == 3:
                df_reference_stations_d_r = df_reference_stations
                list_index.append(df_reference_stations.index[0])
                list_distance_nearest.append(df_reference_stations['distance'].iloc[0])
                
            
            # print(df_reference_stations)
        df_reference_stations.index.name = 'ams'
        df_reference_stations = pd.concat([df_reference_stations_u_l, df_reference_stations_u_r, df_reference_stations_d_l, df_reference_stations_d_r])        
            
    if len(list_index) == 0:
        print('No reference stations found')
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, '-', '-', '-', '-')
    elif len(list_index) == 1:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, int(list_index[0]), '-', '-', '-')
    elif len(list_index) == 2:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, int(list_index[0]), int(list_index[1]), '-', '-')
    elif len(list_index) == 3:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, int(list_index[0]), int(list_index[1]), int(list_index[2]), '-')
    elif len(list_index) == 4:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, int(list_index[0]), int(list_index[1]), int(list_index[2]), int(list_index[3]))

    # check if primary stations are in the range of the secondary station
    
    distance_lon = 10000 # 6400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS
    distance_lat = 10000 # 5400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS

    for i in range(len(loc_prim)):
        lon_check = False
        lat_check = False

        if (loc_prim['lon'][i] < (loc_sec['lon'][station] + distance_lon)) and (loc_prim['lon'][i] > (loc_sec['lon'][station] - distance_lon)):
            lon_check = True
        if (loc_prim['lat'][i] < (loc_sec['lat'][station] + distance_lat)) and (loc_prim['lat'][i] > (loc_sec['lat'][station] - distance_lat)):
            lat_check = True
        if lon_check and lat_check:
            # print('lon:', coordinates_primary_utm32['lon'][i])
            # print('lat:', coordinates_primary_utm32['lat'][i])
            print('primary station', i, 'in range of secondary station', station)
            print('distance to secondary station:', round(np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station])**2), 2))

    return df_reference_stations, list_index, list_distance_nearest

def find_4_nearest_reference_stations_with_plot_and_primary_station(loc_prim, loc_sec, station):
    coordinates = loc_sec
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    # set frame for search
    radius = 1500

    # set coordinates of the station
    lon_station = coordinates['lon'].iloc[station - 1]
    lat_station = coordinates['lat'].iloc[station - 1]

    list_reference_stations_lon = []
    list_reference_stations_lat = []
    list_station = []
    list_distance = []

    # find the 4 nearest stations in frame
    for i in range(len(coordinates)):
        lon = coordinates['lon'].iloc[i]
        lat = coordinates['lat'].iloc[i]
        if (np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2) <= radius):
            if lon == lon_station and lat == lat_station:
                pass
            else:
                # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                # print('\n')

                distance = round(np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2), 2)

                list_distance.append(distance)
                list_station.append(i + 1)
                list_reference_stations_lon.append(round(lon, 2))
                list_reference_stations_lat.append(round(lat, 2))
        
        array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
        df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

# die 4 nächstgelegenen Stationen herausfiltern
        
    if len(df_reference_stations) == 0:
        print('No reference stations found')
        pass
    elif len(df_reference_stations) <= 4:
        for i in range(len(df_reference_stations)):    
            # print(distance)
            pass
    elif len(df_reference_stations) > 4:
        
        list_distance_nearest_reference_stations = []

        for i in range(len(list_distance)):
            if len(list_distance_nearest_reference_stations) < 4:
                index_min_distance = np.argmin(list_distance)
                list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                del list_distance[index_min_distance]

                
        # print(list_distance_nearest_reference_stations)

        for i in df_reference_stations['distance']:
            count = 0
            for j in list_distance_nearest_reference_stations:
                if i == j:
                    count += 1
                    break
                else:
                    continue
            if count == 0:
                df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)
        
    fct.coordinates(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], df_reference_stations.index[1], df_reference_stations.index[2], df_reference_stations.index[3])

    # check if primary stations are in the range of the secondary station
    
    radius = 1500

    for i in range(len(loc_prim)):
        if (np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station])**2) <= radius):
            # print('lon:', coordinates_primary_utm32['lon'][i])
            # print('lat:', coordinates_primary_utm32['lat'][i])
            print('primary station', i, 'in range of secondary station', station)
            print('distance to secondary station:', round(np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station])**2), 2))


    return df_reference_stations #, list_distance_nearest_reference_stations

def berechnung_Gewichte_für_4_nearest(list_distance_nearest):
    
    list_weights = []

    distance1 = list_distance_nearest.iloc[0]
    distance2 = list_distance_nearest.iloc[1]
    distance3 = list_distance_nearest.iloc[2]
    distance4 = list_distance_nearest.iloc[3]

    W1 = (1/distance1**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
    W2 = (1/distance2**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
    W3 = (1/distance3**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
    W4 = (1/distance4**2)/((1/distance1**2)+(1/distance2**2)+(1/distance3**2)+(1/distance4**2))
    list_weights.append(W1)
    list_weights.append(W2)
    list_weights.append(W3)
    list_weights.append(W4)

    return list_weights

def berechnung_Referenzniederschlag_4_nearest(secondary_data_df, secondary_data_df_nonan, df_reference_values, station, list_weights, list_index):
    list_counts_start, list_counts_end, list_counts, list_index_peak = fct.get_data_nan_seq_before_peak(secondary_data_df, 'sc', station, 0.99)

    for c, p in zip(range(0, len(list_counts_start)), range(0, len(list_index_peak))):    
        for index in secondary_data_df_nonan.loc[list_counts_start[c] : list_index_peak[p]].index:
            h1 = secondary_data_df_nonan['ams' + str(list_index[0])].loc[index]
            h2 = secondary_data_df_nonan['ams' + str(list_index[1])].loc[index]
            h3 = secondary_data_df_nonan['ams' + str(list_index[2])].loc[index]
            h4 = secondary_data_df_nonan['ams' + str(list_index[3])].loc[index]

            h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3 + list_weights[3]*h4

            df_reference_values.loc[index] = h_ref

    return df_reference_values

def calculate_correlation_with_without_resample(data_primary, data_secondary, prim_station, sec_station, percentile, resample_sec, resample_prim):
    if resample_sec == True:
        df_agg = fct.resampleDf(data_secondary[sec_station], 'H', closed='right', label='right', shift=False, leave_nan=True, label_shift=None, temp_shift=0, max_nan=2)
    else:
        df_agg = data_secondary[sec_station]

    index_start = data_secondary.index[0]
    index_end = data_secondary.index[-1]

    if resample_prim == True:
        df_reference = fct.resampleDf(data_primary[prim_station][index_start:index_end], 'H', closed='right', label='right', shift=False, leave_nan=True, label_shift=None, temp_shift=0, max_nan=2)
    else:
        df_reference = data_primary[prim_station][index_start:index_end]

    df_for_correlation = pd.concat([df_reference, df_agg], axis=1)
    df_for_correlation = df_for_correlation.dropna() # ohne wäre correlation nan

    cc, a_dataset, b_dataset = fct.calc_indicator_correlation(df_for_correlation.iloc[:, 0].values, df_for_correlation.iloc[:, 1].values, percentile)

    return cc

# def correction_complete_amsterdam(sec_utm, station_zahl, secondary_data_df_nonan, secondary_data_df):

#     '''
#     0.99 Perzentil bei find nan sequence before peak
#     5 km max Entfernung zu den Referenzstationen
#     '''

#     reference_stations = fct.find_4_nearest_reference_stations(sec_utm, station_zahl)
#     list_weights = fct.berechnung_Gewichte_für_4_nearest(list_distance_nearest=reference_stations['distance'])
#     df_reference_values = secondary_data_df[['ams' + str(station_zahl)]].copy()
#     df_reference_values_calculated = fct.berechnung_Referenzniederschlag_4_nearest(secondary_data_df, secondary_data_df_nonan, df_reference_values, list_weights, reference_stations.index)
#     data_corrected = fct.correct_data(secondary_data_df, df_reference_values_calculated, 'sc', 'ams' + str(station_zahl), 0.99)

#     return data_corrected

def correction_complete_amsterdam(coordinates_secondary_utm32, station_zahl, secondary_data_df_nonan, secondary_data_df):

    reference_stations = fct.find_4_nearest_reference_stations(coordinates_secondary_utm32, station_zahl)
    if len(reference_stations) == 0:
        return
    list_weights = fct.berechnung_Gewichte(reference_stations.index, reference_stations['distance'])
    df_reference_values = secondary_data_df[['ams' + str(station_zahl)]].copy()
    df_reference_values_calculated = fct.berechnung_Referenzniederschlag_amsterdam(secondary_data_df_nonan, secondary_data_df, df_reference_values, list_weights, reference_stations.index, station_zahl)
    data_corrected = fct.correct_data_new(secondary_data_df, df_reference_values_calculated, 'sc', 'ams' + str(station_zahl), 0.99, True, True, True)

    return data_corrected

def coordinates_with_find_primary(loc_prim, loc_sec, y, station, ref1, ref2, ref3, ref4, prim_in_sec=False):
    
    if y == 'primary':
        coords_lon = loc_prim['lon']
        coords_lat = loc_prim['lat']
    elif y == 'secondary':
        coords_lon = loc_sec['lon']
        coords_lat = loc_sec['lat']
    elif y == 'both':
        coords_lon_prim = loc_prim['lon']
        coords_lat_prim = loc_prim['lat']
        coords_lon_sec = loc_sec['lon']
        coords_lat_sec = loc_sec['lat']

    if y == 'both':
        name_plot = 'Coordinates ' + y + ' networks'
        plt.scatter(x=coords_lon_prim, y=coords_lat_prim, s=20, color='red', label='primary network', marker='x', linewidth=1)
        plt.scatter(x=coords_lon_sec, y=coords_lat_sec, s=2, color='blue', label='secondary network', alpha=0.5)
        if type(station) == int:
            plt.scatter(loc_prim['lon'].iloc[station], loc_prim['lat'].iloc[station], color='black')
        plt.legend()
    else:
        name_plot = 'Coordinates ' + y + ' network'
        plt.scatter(x=coords_lon, y=coords_lat, s=10)
        if type(station) == int:
            if y == 'primary':
                plt.scatter(loc_prim['lon'].iloc[station], loc_prim['lat'].iloc[station], color='red')
            elif y == 'secondary':
                plt.scatter(loc_sec['lon'].iloc[station - 1], loc_sec['lat'].iloc[station - 1], color='red')

                if prim_in_sec == True:
                    # check if primary stations are in the range of the secondary station
                        
                        radius = 3000

                        for i in range(len(loc_prim)):
                            
                            if (np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station])**2) <= radius):

                                # print('lon:', coordinates_primary_utm32['lon'][i])
                                # print('lat:', coordinates_primary_utm32['lat'][i])
                                print('primary station', i, 'in range of secondary station', station)
                                print('distance to secondary station:', round(np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station])**2), 2))

                                plt.scatter(loc_prim['lon'].iloc[i], loc_prim['lat'].iloc[i], color='black', marker='x', linewidth=1)

            try:
                plt.scatter(loc_sec['lon'].iloc[ref1 - 1], loc_sec['lat'].iloc[ref1 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref2 - 1], loc_sec['lat'].iloc[ref2 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref3 - 1], loc_sec['lat'].iloc[ref3 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref4 - 1], loc_sec['lat'].iloc[ref4 - 1], color='lime', s=10)
            except:
                pass

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(name_plot)

    plt.show()
    plt.close()
    
    return # print(coords_lon, coords_lat)

def correction_complete_amsterdam_with_primary(sec_utm, prim_utm, station_zahl, secondary_data_df_nonan, secondary_data_df, primary_data_df):
       
    # check if primary stations are in the range of the secondary station
    
    distance_lon = 5000 # 6400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS
    distance_lat = 5000 # 5400 entspricht ca. einem Viertel der Ausbreitungsfläche der PWS

    count = 0

    for i in range(len(prim_utm)):
        lon_check = False
        lat_check = False

        if (prim_utm['lon'][i] < (sec_utm['lon'][station_zahl] + distance_lon)) and (prim_utm['lon'][i] > (sec_utm['lon'][station_zahl] - distance_lon)):
            lon_check = True
        if (prim_utm['lat'][i] < (sec_utm['lat'][station_zahl] + distance_lat)) and (prim_utm['lat'][i] > (sec_utm['lat'][station_zahl] - distance_lat)):
            lat_check = True
        if lon_check and lat_check:
            df_reference_values = primary_data_df[[i]].copy().rename(columns={i : 'ams' + str(station_zahl)})
            count += 1

    if count == 0:
        reference_stations = fct.find_4_nearest_reference_stations(sec_utm, station_zahl)
        list_weights = fct.berechnung_Gewichte_für_4_nearest(list_distance_nearest=reference_stations['distance'])
        df_reference_values = secondary_data_df[['ams' + str(station_zahl)]].copy()
        df_reference_values_calculated = fct.berechnung_Referenzniederschlag_4_nearest(secondary_data_df, secondary_data_df_nonan, df_reference_values, list_weights, reference_stations.index)
        data_corrected = fct.correct_data(secondary_data_df, df_reference_values_calculated, 'sc', 'ams' + str(station_zahl), 0.99)

    if count != 0:
        secondary_data_df_res = fct.resampleDf(secondary_data_df[['ams' + str(station_zahl)]], 'H', closed='right', label='right', shift=False, leave_nan=True, label_shift=None, temp_shift=0, max_nan=2)
        
        index_start = secondary_data_df[['ams' + str(station_zahl)]].index[0]
        index_end = secondary_data_df[['ams' + str(station_zahl)]].index[-1]

        df_reference_values_correct_index = df_reference_values[index_start:index_end]

        data_corrected = fct.correct_data(secondary_data_df_res, df_reference_values_correct_index, 'pr', 'ams' + str(station_zahl), 0.99)   

    return data_corrected

def list_nan_sequences(dataframe, column, min_len):

    dataframe = dataframe
    dataframe_mask = dataframe.isna()
    column = column

    list = []

    count = 0
    index_count = 0

    for value in dataframe_mask[column]:
        if value == True:
            if count == 0:    
                index_start = dataframe[column].index[index_count]
                count += 1
            else:
                count += 1
                if dataframe[column].index[index_count] == dataframe[column].index[-1]:
                    index_end = dataframe[column].index[index_count]
                    if count < min_len:
                        pass
                    else:
                        list.append([count, index_start, index_end])

        else:
            if count == 0:
                pass
            else:
                index_end = dataframe[column].index[index_count - 1]
                if count < min_len:
                    pass
                else:
                    list.append(count) #, index_start, index_end])
                count = 0
        index_count += 1

    print('Count of nan sequences with min len ' + str(min_len) + ':', len(list))

    return list

def coordinates_reutlingen(loc_prim, loc_sec, y, station, ref1, ref2, ref3, ref4):
    
    if y == 'primary':
        coords_lon = loc_prim['lon']
        coords_lat = loc_prim['lat']
    elif y == 'secondary':
        coords_lon = loc_sec['lon']
        coords_lat = loc_sec['lat']
    elif y == 'both':
        coords_lon_prim = loc_prim['lon']
        coords_lat_prim = loc_prim['lat']
        coords_lon_sec = loc_sec['lon']
        coords_lat_sec = loc_sec['lat']

    if y == 'both':
        name_plot = 'Coordinates ' + y + ' networks'
        plt.scatter(x=coords_lon_prim, y=coords_lat_prim, s=20, color='red', label='primary network', marker='x', linewidth=1)
        plt.scatter(x=coords_lon_sec, y=coords_lat_sec, s=2, color='blue', label='secondary network', alpha=0.5)
        if type(station) == int:
            plt.scatter(loc_prim['lon'].iloc[station - 1], loc_prim['lat'].iloc[station - 1], color='black')
            # print('------ check 1 ------')
            # print('lon:', loc_prim['lon'].iloc[station - 1], 'lat:', loc_prim['lat'].iloc[station - 1])
        plt.legend()
    else:
        name_plot = 'Coordinates ' + y + ' network'
        plt.scatter(x=coords_lon, y=coords_lat, s=10)
        if type(station) == int:
            if y == 'primary':
                plt.scatter(loc_prim['lon'].iloc[station - 1], loc_prim['lat'].iloc[station - 1], color='red')
            elif y == 'secondary':
                plt.scatter(loc_sec['lon'].iloc[station - 1], loc_sec['lat'].iloc[station - 1], color='red')
            # print('------ check 2 ------')
            # print('lon:', loc_sec['lon'].iloc[station - 1], 'lat:', loc_sec['lat'].iloc[station - 1])

            try:
                plt.scatter(loc_sec['lon'].iloc[ref1 - 1], loc_sec['lat'].iloc[ref1 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref2 - 1], loc_sec['lat'].iloc[ref2 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref3 - 1], loc_sec['lat'].iloc[ref3 - 1], color='lime', s=10)
                plt.scatter(loc_sec['lon'].iloc[ref4 - 1], loc_sec['lat'].iloc[ref4 - 1], color='lime', s=10)
            except:
                pass

    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(name_plot)

    plt.show()
    plt.close()
    
    return # print(coords_lon, coords_lat)

def find_4_nearest_reference_stations_with_plot_and_primary_station_reutlingen(loc_prim, loc_sec, station):
    coordinates = loc_sec
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    # set frame for search

    radius = 1500

    # set coordinates of the station
    lon_station = coordinates['lon'].iloc[station - 1]
    lat_station = coordinates['lat'].iloc[station - 1]

    # print('------ check 1 ------')
    # print('lon:', lon_station, 'lat:', lat_station)

    list_reference_stations_lon = []
    list_reference_stations_lat = []
    list_station = []
    list_distance = []

    # find the 4 nearest stations in frame
    for i in range(len(coordinates)):
        lon = coordinates['lon'].iloc[i]
        lat = coordinates['lat'].iloc[i]
        if (np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2) <= radius):
            if lon == lon_station and lat == lat_station:
                pass
            else:
                # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                # print('\n')

                distance = round(np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2), 2)

                list_distance.append(distance)
                list_station.append(i + 1)
                list_reference_stations_lon.append(round(lon, 2))
                list_reference_stations_lat.append(round(lat, 2))
        
        array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
        df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

# die 4 nächstgelegenen Stationen herausfiltern
        
    if len(df_reference_stations) == 0:
        print('No reference stations found')
        pass
    elif len(df_reference_stations) <= 4:
        for i in range(len(df_reference_stations)):    
            # print(distance)
            pass
    elif len(df_reference_stations) > 4:
        
        list_distance_nearest_reference_stations = []

        for i in range(len(list_distance)):
            if len(list_distance_nearest_reference_stations) < 4:
                index_min_distance = np.argmin(list_distance)
                list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                del list_distance[index_min_distance]

                
        # print(list_distance_nearest_reference_stations)

        for i in df_reference_stations['distance']:
            count = 0
            for j in list_distance_nearest_reference_stations:
                if i == j:
                    count += 1
                    break
                else:
                    continue
            if count == 0:
                df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)
        
    if len(df_reference_stations) == 0:
        print('No reference stations found')
        pass
    elif len(df_reference_stations) == 1:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], '-', '-', '-')
    elif len(df_reference_stations) == 2:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], df_reference_stations.index[1], '-', '-')
    elif len(df_reference_stations) == 3:
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], df_reference_stations.index[1], df_reference_stations.index[2], '-')
    elif len(df_reference_stations) >= 4:        
        fct.coordinates(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], df_reference_stations.index[1], df_reference_stations.index[2], df_reference_stations.index[3])

    # check if primary stations are in the range of the secondary station
    
    radius = 1500

    for i in range(1, len(loc_prim) + 1):
        if (np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station - 1])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station - 1])**2) <= radius):
            print('lon:', loc_prim['lon'][i])
            print('lat:', loc_prim['lat'][i])
            print('primary station', i, 'in range of secondary station', station)
            print('distance to secondary station:', round(np.sqrt((loc_prim['lon'][i] - loc_sec['lon'][station - 1])**2 + (loc_prim['lat'][i] - loc_sec['lat'][station - 1])**2), 2))
            # print('------ check 3 ------')
            # print('lon:', loc_prim['lon'][i], 'lat:', loc_prim['lat'][i])

    # print('------ check 2 ------')
    # print('lon:', loc_sec['lon'][station - 1], 'lat:', loc_sec['lat'][station - 1])

    

    return df_reference_stations #, list_distance_nearest_reference_stations

def find_pluvio_reference_stations(prim_utm, sec_utm, station):
    
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    # set frame for search

    radius = 1500

    # set coordinates of the station
    lon_station = sec_utm['lon'].iloc[station - 1]
    lat_station = sec_utm['lat'].iloc[station - 1]

    list_reference_stations_lon = []
    list_reference_stations_lat = []
    list_station = []
    list_distance = []

    # find the 4 nearest stations in frame
    for i in range(len(prim_utm)):
        lon = prim_utm['lon'].iloc[i]
        lat = prim_utm['lat'].iloc[i]
        if (np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2) <= radius):
            if lon == lon_station and lat == lat_station:
                pass
            else:
                # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                # print('\n')

                distance = np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2)

                list_distance.append(distance)
                list_station.append(i + 1)
                list_reference_stations_lon.append(lon)
                list_reference_stations_lat.append(lat)
        
        array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
        df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

# die 4 nächstgelegenen Stationen herausfiltern
        
    if len(df_reference_stations) == 0:
        print('No reference stations found')
        pass
    elif len(df_reference_stations) <= 4:
        for i in range(len(df_reference_stations)):    
            # print(distance)
            pass
    elif len(df_reference_stations) > 4:
        
        list_distance_nearest_reference_stations = []

        for i in range(len(list_distance)):
            if len(list_distance_nearest_reference_stations) < 4:
                index_min_distance = np.argmin(list_distance)
                list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                del list_distance[index_min_distance]

                
        # print(list_distance_nearest_reference_stations)

        for i in df_reference_stations['distance']:
            count = 0
            for j in list_distance_nearest_reference_stations:
                if i == j:
                    count += 1
                    break
                else:
                    continue
            if count == 0:
                df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)

    return df_reference_stations




def find_pluvio_reference_stations_reutlingen(prim_utm, sec_utm, station):
    
# finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    # set frame for search
    
    radius = 1500

    # set coordinates of the station
    lon_station = sec_utm['lon'].iloc[station - 1]
    lat_station = sec_utm['lat'].iloc[station - 1]

    list_reference_stations_lon = []
    list_reference_stations_lat = []
    list_station = []
    list_distance = []

    # find the 4 nearest stations in frame
    for i in range(len(prim_utm)):
        lon = prim_utm['lon'].iloc[i]
        lat = prim_utm['lat'].iloc[i]
        if (np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2) <= radius):
            if lon == lon_station and lat == lat_station:
                pass
            else:
                # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                # print('\n')

                distance = np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2)

                list_distance.append(distance)
                list_station.append(i + 1)
                list_reference_stations_lon.append(lon)
                list_reference_stations_lat.append(lat)
        
        array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
        df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

# die 4 nächstgelegenen Stationen herausfiltern
        
    if len(df_reference_stations) == 0:
        print('No reference stations found')
        pass
    elif len(df_reference_stations) <= 4:
        for i in range(len(df_reference_stations)):    
            # print(distance)
            pass
    elif len(df_reference_stations) > 4:
        
        list_distance_nearest_reference_stations = []

        for i in range(len(list_distance)):
            if len(list_distance_nearest_reference_stations) < 4:
                index_min_distance = np.argmin(list_distance)
                list_distance_nearest_reference_stations.append(list_distance[index_min_distance])
                del list_distance[index_min_distance]

                
        # print(list_distance_nearest_reference_stations)

        for i in df_reference_stations['distance']:
            count = 0
            for j in list_distance_nearest_reference_stations:
                if i == j:
                    count += 1
                    break
                else:
                    continue
            if count == 0:
                df_reference_stations = df_reference_stations.drop(index=df_reference_stations[df_reference_stations['distance'] == i].index)

    return df_reference_stations

# def berechnung_Referenzniederschlag_reutlingen_pluvio(primary_data_df_nonan, secondary_data_df, df_reference_values, list_weights, list_index, name_station_str):

#     list_counts_start, list_counts_end, list_counts, list_index_peak = fct.get_data_nan_seq_before_peak(secondary_data_df, 'pr', name_station_str, 0.99)

#     for c, p in zip(range(0, len(list_counts_start)), range(0, len(list_index_peak))):    
#         for index in primary_data_df_nonan[primary_data_df_nonan['id'] == 1].loc[list_counts_start[c] : list_index_peak[p]].index:
#             if len(list_index) == 1:
#                 h1 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[0]].loc[index]
#                 h_ref = list_weights[0]*h1
#             elif len(list_index) == 2:
#                 h1 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[0]].loc[index]
#                 h2 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[1]].loc[index]
#                 h_ref = list_weights[0]*h1 + list_weights[1]*h2
#             elif len(list_index) == 3:
#                 h1 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[0]].loc[index]
#                 h2 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[1]].loc[index]
#                 h3 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[2]].loc[index]
#                 h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3
#             elif len(list_index) == 4:
#                 h1 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[0]].loc[index]
#                 h2 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[1]].loc[index]
#                 h3 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[2]].loc[index]
#                 h4 = primary_data_df_nonan[primary_data_df_nonan['id'] == list_index[3]].loc[index]
#                 h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3 + list_weights[3]*h4

#             df_reference_values.loc[index] = h_ref

#     return df_reference_values

def berechnung_Referenzniederschlag_reutlingen_pluvio(secondary_data_df_nonan, secondary_data_df, df_reference_values, list_weights, list_index, name_station_str):

    list_counts_start, list_counts_end, list_counts, list_index_peak = fct.get_data_nan_seq_before_peak(secondary_data_df, 'sc', name_station_str, 0.99)

    for i in range(0, len(list_index)):
        if list_index[i] < 10:
            list_index[i] = 'RT_0' + str(list_index[i])
        elif list_index[i] < 100 and list_index[i] >= 10:
            list_index[i] = 'RT_' + str(list_index[i])
            
    for c, p in zip(range(0, len(list_counts_start)), range(0, len(list_index_peak))):    
        for index in secondary_data_df_nonan.loc[list_counts_start[c] : list_index_peak[p]].index:
            if len(list_index) == 1:
                h1 = secondary_data_df_nonan[list_index[0]].loc[index]
                h_ref = list_weights[0]*h1
            elif len(list_index) == 2:
                h1 = secondary_data_df_nonan[list_index[0]].loc[index]
                h2 = secondary_data_df_nonan[list_index[1]].loc[index]
                h_ref = list_weights[0]*h1 + list_weights[1]*h2
            elif len(list_index) == 3:
                h1 = secondary_data_df_nonan[list_index[0]].loc[index]
                h2 = secondary_data_df_nonan[list_index[1]].loc[index]
                h3 = secondary_data_df_nonan[list_index[2]].loc[index]
                h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3
            elif len(list_index) == 4:
                h1 = secondary_data_df_nonan[list_index[0]].loc[index]
                h2 = secondary_data_df_nonan[list_index[1]].loc[index]
                h3 = secondary_data_df_nonan[list_index[2]].loc[index]
                h4 = secondary_data_df_nonan[list_index[3]].loc[index]
                h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3 + list_weights[3]*h4

            df_reference_values.loc[index] = h_ref

    return df_reference_values

def berechnung_Referenzniederschlag_reutlingen_pws(secondary_data_df_nonan, secondary_data_df, df_reference_values, list_weights, list_index, name_station_str):

    list_counts_start, list_counts_end, list_counts, list_index_peak = fct.get_data_nan_seq_before_peak(secondary_data_df, 'sc', name_station_str, 0.99)
    
    for i in range(0, len(list_index)):
        if list_index[i] < 10:
            list_index[i] = 'RT_00' + str(list_index[i])
        elif list_index[i] < 100 and list_index[i] >= 10:
            list_index[i] = 'RT_0' + str(list_index[i])
        elif list_index[i] >= 100:
            list_index[i] = 'RT_' + str(list_index[i])
            
    for c, p in zip(range(0, len(list_counts_start)), range(0, len(list_index_peak))):    
        for index in secondary_data_df_nonan.loc[list_counts_start[c] : list_index_peak[p]].index:
            if len(list_index) == 1:
                h1 = secondary_data_df_nonan[list_index[0]].loc[index]
                h_ref = list_weights[0]*h1
            elif len(list_index) == 2:
                h1 = secondary_data_df_nonan[list_index[0]].loc[index]
                h2 = secondary_data_df_nonan[list_index[1]].loc[index]
                h_ref = list_weights[0]*h1 + list_weights[1]*h2
            elif len(list_index) == 3:
                h1 = secondary_data_df_nonan[list_index[0]].loc[index]
                h2 = secondary_data_df_nonan[list_index[1]].loc[index]
                h3 = secondary_data_df_nonan[list_index[2]].loc[index]
                h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3
            elif len(list_index) == 4:
                h1 = secondary_data_df_nonan[list_index[0]].loc[index]
                h2 = secondary_data_df_nonan[list_index[1]].loc[index]
                h3 = secondary_data_df_nonan[list_index[2]].loc[index]
                h4 = secondary_data_df_nonan[list_index[3]].loc[index]
                h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3 + list_weights[3]*h4

            df_reference_values.loc[index] = h_ref

    return df_reference_values

def correction_complete_reutlingen_with_primary(sec_utm, prim_utm, station_zahl, secondary_data_df_nonan, secondary_data_df, primary_data_df_nonan):
    
    radius = 1500
    
    # transform station number to string

    if station_zahl < 10:
        name_station_str = 'RT_00' + str(station_zahl)
    elif station_zahl >= 10 and station_zahl < 100:
        name_station_str = 'RT_0' + str(station_zahl)
    elif station_zahl >= 100:
        name_station_str = 'RT_' + str(station_zahl)

    # check if primary stations are in range of the secondary station

    count = 0

    for i in range(1, len(prim_utm) + 1):

        if (np.sqrt((prim_utm['lon'][i] - sec_utm['lon'][station_zahl - 1])**2 + (prim_utm['lat'][i] - sec_utm['lat'][station_zahl - 1])**2) <= radius):
            count += 1
            break

    if count != 0:

        print('Primary station found')
        reference_stations = fct.find_pluvio_reference_stations_reutlingen(prim_utm, sec_utm, station_zahl)
        list_index = reference_stations.index.tolist()    
        list_weights = fct.berechnung_Gewichte(list_index, list_distance_nearest=reference_stations['distance'])
        df_reference_values = secondary_data_df[[name_station_str]].copy()
        df_reference_values_calculated = fct.berechnung_Referenzniederschlag_reutlingen_pluvio(primary_data_df_nonan, secondary_data_df, df_reference_values, list_weights, list_index, name_station_str)
        data_corrected = fct.correct_data_new(secondary_data_df, df_reference_values_calculated, 'sc', name_station_str, 0.99) 
        
    if count == 0:

        print('No primary station found')
        reference_stations = fct.find_4_nearest_reference_stations(sec_utm, station_zahl)
        list_index = reference_stations.index.tolist()

        if len(list_index) == 0:
            return
        
        list_weights = fct.berechnung_Gewichte(list_index, list_distance_nearest=reference_stations['distance'])
        df_reference_values = secondary_data_df[[name_station_str]].copy()
        df_reference_values_calculated = fct.berechnung_Referenzniederschlag_reutlingen_pws(secondary_data_df_nonan, secondary_data_df, df_reference_values, list_weights, list_index, name_station_str)
        data_corrected = fct.correct_data_new(secondary_data_df, df_reference_values_calculated, 'sc', name_station_str, 0.99)

    return data_corrected

def histogramm_scatter(data, column, timedelta):

    station = fct.list_nan_sequences_schnell(data, column, timedelta)[2]

    absolute_frequencies = np.unique(station, return_counts=True)

    x = absolute_frequencies[0]
    y = absolute_frequencies[1]

    hist = np.histogram(station, bins=[10**i for i in range(7)])[0]

    # plot
    fig, ax = plt.subplots()

    ax.scatter(x, y, marker='.', alpha=0.5, edgecolors='none', s=100)

    plt.hlines(hist[0], 1, 10, color='r', linestyle='--', label='absolute Häufigkeit im Intervall 10**i')
    plt.hlines(hist[1], 10, 100, color='r', linestyle='--')
    plt.hlines(hist[2], 100, 1000, color='r', linestyle='--')
    plt.hlines(hist[3], 1000, 10000, color='r', linestyle='--')
    plt.hlines(hist[4], 10000, 100000, color='r', linestyle='--')
    plt.hlines(hist[5], 100000, 1000000, color='r', linestyle='--')

    plt.vlines(1, 0, hist[0], color='r', linestyle='--')
    plt.vlines(10, 0, hist[0], color='r', linestyle='--')
    plt.vlines(10, 0, hist[1], color='r', linestyle='--')
    plt.vlines(100, 0, hist[1], color='r', linestyle='--')
    plt.vlines(100, 0, hist[2], color='r', linestyle='--')
    plt.vlines(1000, 0, hist[2], color='r', linestyle='--')
    plt.vlines(1000, 0, hist[3], color='r', linestyle='--')
    plt.vlines(10000, 0, hist[3], color='r', linestyle='--')
    plt.vlines(10000, 0, hist[4], color='r', linestyle='--')
    plt.vlines(100000, 0, hist[4], color='r', linestyle='--')
    plt.vlines(100000, 0, hist[5], color='r', linestyle='--')
    plt.vlines(1000000, 0, hist[5], color='r', linestyle='--')

    plt.xscale('log')
    ticks = [10**i for i in range(6)]  # Erzeugt eine Liste [1, 10, 100, 1000]
    plt.xticks(ticks, labels=ticks)  # Setzt die x-Achsenwerte und -Beschriftungen

    plt.yscale('log')
    plt.yticks(ticks, labels=ticks)  # Setzt die y-Achsenwerte und -Beschriftungen

    plt.xlabel('Länge der nan-Sequenz')
    plt.ylabel('Absolute Häufigkeit')
    try:
        plt.title('Längen und absolute Häufigkeiten der nan-Sequenzen, Station: ' + column)
    except:
        plt.title('Längen und absolute Häufigkeiten der nan-Sequenzen, Station: ' + str(column))

    plt.legend()

    plt.show()

    return

def list_nan_sequences_schnell(data, station, timedelta):
    
    '''starts, ends, len_seq'''
    
    if timedelta == '1min':
        timedelta = datetime.timedelta(minutes=1)
    if timedelta == '5min':
        timedelta = datetime.timedelta(minutes=5)
    if timedelta == '1h':
        timedelta = datetime.timedelta(hours=1)

    is_nan = data[station].isna() # gibt true zurück, wenn Wert NaN ist
    diff = is_nan.diff() # gibt true zurück, wenn Wert zu Nan oder Nan zu Wert springt

    # print(is_nan)
    # print(diff)

    if is_nan[0] == True:
        diff[0] = True

    starts = diff[diff == True].index[::2]
    ends = diff[diff == True].index[1::2] - timedelta

    if is_nan[-1] == True:
        ends = ends.append(data.index[-1:])
    elif len(starts) > len(ends):
        starts = starts.delete(-1)

    len_seq = ((ends + timedelta) - starts)/timedelta
    len_seq = len_seq.astype(int)
    
    # print('starts:', starts)
    # print('ends:', ends)
    
    return starts, ends, len_seq

def einer_zweier_sequ_korrigieren(data, column, _1seq=True, _2seq=True):
    # nan die alleine Stehen oder nan-Paare zu 0 machen

    station_touse = data

    station = station_touse[[column]].copy()

    # True für alleinstehende nans
    mask_1seq = station.isna() & station.shift(-1).notnull() & station.shift(1).notnull()

    # erste und letzte Werte extra betrachten
    if (station.iloc[0].isna() & station.iloc[1].notnull()).bool():
        mask_1seq.iloc[0] = True
    if (station.iloc[-1].isna() & station.iloc[-2].notnull()).bool():
        mask_1seq.iloc[-1] = True

    # True für nan-Paare
    mask1 = station.isna() & station.shift(-1).notnull() & station.shift(1).isna() & station.shift(2).notnull()
    mask2 = station.isna() & station.shift(-1).isna() & station.shift(1).notnull() & station.shift(-2).notnull()
    maske_2seq = mask1 | mask2

    # erste zwei und letzte zwei Werte extra betrachten
    if (station.iloc[0].isna() & station.iloc[1].isna() & station.iloc[2].notnull()).bool():
        maske_2seq.iloc[0] = True
        maske_2seq.iloc[1] = True
    if (station.iloc[-1].isna() & station.iloc[-2].isna() & station.iloc[-3].notnull()).bool():
        maske_2seq.iloc[-1] = True
        maske_2seq.iloc[-2] = True

    if _1seq == True:
        station[mask_1seq] = 0
    if _2seq == True:
        station[maske_2seq] = 0
    
    return station

def correct_data_new(data, reference, y, station, quantile, correct_peak=True, correct_1_2=True, correct_0_pres_ref=True):

    # nans vor peaks korrigieren
    if correct_peak:

        output_list_counts_start, output_list_counts_end, output_list_counts, output_list_index_peak = fct.get_data_nan_seq_before_peak(data, y, station, quantile)
    
        data_corrected = data[[station]].copy() # copy the data to a new dataframe

        if y == 'pr':
            frequency = '1h'
        elif y == 'sc':
            frequency = '5min'

        for i in range(len(output_list_index_peak)):

            datetime_index = pd.date_range(start=output_list_counts_start[i], end=output_list_index_peak[i], freq=frequency) # create a datetime index for the time period of the nan sequence before the peak
            sum = reference[station].loc[output_list_counts_start[i] : output_list_index_peak[i]].sum() # sum of the reference values for the time period of the nan sequence before the peak
            value_peak = data[station].loc[output_list_index_peak[i]] # value of the peak

            for index in datetime_index:
                try:
                    peak_portion = round(((reference[station].loc[index] / sum) * value_peak), 2)
                except ZeroDivisionError:
                    peak_portion = 0
                    
                data_corrected[station].loc[index] = peak_portion # replace the nan values with the calculated peak portion
            
    # 1er und 2er nan sequenzen korrigieren
    if correct_1_2:

        data_corrected = fct.einer_zweier_sequ_korrigieren(data_corrected, station, True, True)

    # nan sequenzen korrigieren, die mit 0 anfnagen und enden und bei denen die summe des niederschlag im referenz df 0 ist
    if correct_0_pres_ref:

        starts, ends, nan_sequs = fct.list_nan_sequences_schnell(data_corrected, station, frequency)

        list = []

        # erste und letzte nan sequenz extra kontrollieren
        sum = reference[station].loc[starts[0] : ends[0]].values.sum()
        if sum == 0:
            list.append(0)
        sum = reference[station].loc[starts[-1] : ends[-1]].values.sum()
        if sum == 0:
            list.append(-1)

        for i in range(len(starts)):
            if y == 'sc':
                try:
                    if ((data_corrected.loc[starts[i] - datetime.timedelta(minutes=5)]) == 0).bool() and ((data_corrected.loc[ends[i] + datetime.timedelta(minutes=5)]) == 0).bool():
                        sum = reference[station].loc[starts[i] : ends[i]].values.sum()
                        if sum == 0:
                            list.append(i)
                except KeyError:
                    continue
            elif y == 'pr':
                try:
                    if ((data_corrected.loc[starts[i] - datetime.timedelta(hours=1)]) == 0).bool() and ((data_corrected.loc[ends[i] + datetime.timedelta(hours=1)]) == 0).bool():
                        sum = reference[station].loc[starts[i] : ends[i]].values.sum()
                        if sum == 0:
                            list.append(i)
                except KeyError:
                    continue
            
        for i in list:
            data_corrected.loc[starts[i] : ends[i]] = 0

    return data_corrected

def statistik_nan_sequences_lauchäcker(data, data_reference, column, freq):

    starts, ends, list_nan_sequ = fct.list_nan_sequences_schnell(data, column, freq)
    frequ_nan_sequ  = np.unique(list_nan_sequ, return_counts=True)[1]
    sum_nan_sequ = frequ_nan_sequ.sum()

    try:
        data_corr_1_2 = fct.einer_zweier_sequ_korrigieren(data, column, True, True)
        starts12, ends12, list_nan_sequ_1_2_corr = fct.list_nan_sequences_schnell(data_corr_1_2, column, freq)
        frequ_nan_sequ_1_2_corr  = np.unique(list_nan_sequ_1_2_corr, return_counts=True)[1]
        sum_nan_sequ_1_2_corr = frequ_nan_sequ_1_2_corr.sum()
    except:
        print('Error in correction of 1er and 2er NaN-sequences')

    if freq == '1min':
        print('timedelta error')
        return
    elif freq == '5min':
        y = 'sc'
    elif freq == '1h':
        y = 'pr'

    timestamps_peaks = fct.get_data_nan_seq_before_peak(data, y, column, 0.99)[3]
    sum_peaks = len(timestamps_peaks)

    print('Anzahl der NaN-Sequenzen: ', sum_nan_sequ)
    print('Anzahl der NaN-Sequenzen mit Peak: ', sum_peaks)
    try:
        print('Anzahl der NaN-Sequenzen nach Korrektur von 1er und 2er NaN-Abfolgen: ', sum_nan_sequ_1_2_corr)
    except:
        print('Anzahl der NaN-Sequenzen nach Korrektur von 1er und 2er NaN-Abfolgen: ', 'Error')
    
    list = []
    for i in range(len(starts12)):
        try:
            if ((data_corr_1_2.loc[starts12[i] - datetime.timedelta(minutes=5)]) == 0).bool() and ((data_corr_1_2.loc[ends12[i] + datetime.timedelta(minutes=5)]) == 0).bool():
                list.append(i)
        except KeyError:
            continue

    df_reference = data_reference.rename(columns={data_reference.columns[0] : column})

    list_0_niederschlag = []
    for i in list:
        sum = df_reference.loc[starts12[i] : ends12[i]].values.sum()
        if sum == 0: # niederschlag pro stunde im mittel für den gesamten 0...0 zeitraum
            list_0_niederschlag.append(i)
            # print(i, round(sum/((cs[i]*5)/60), 2))

    timestamps_peaks_12 = fct.get_data_nan_seq_before_peak(data_corr_1_2, y, column, 0.99)[3]
    sum_peaks_12 = len(timestamps_peaks_12)
    
    print('\n')
    print('NaN-Sequenzen mit voraus- und nachgehender 0 nach Korrektur der 1er- und 2er-Sequ.: ', len(list))
    print('     davon Sequenzen, die keinen Niederschlag im Ref.-df haben: ', len(list_0_niederschlag))
    print('NaN-Sequenzen mit Peak nach Korrektur der 1er- und 2er-Sequ.: ', sum_peaks_12)

    return

def berechnung_Referenzniederschlag_amsterdam(secondary_data_df_nonan, secondary_data_df, df_reference_values, list_weights, list_index, station_number):

    list_counts_start, list_counts_end, list_counts, list_index_peak = fct.get_data_nan_seq_before_peak(secondary_data_df, 'sc', 'ams' + str(station_number), 0.99)
            
    for c, p in zip(range(0, len(list_counts_start)), range(0, len(list_index_peak))):    
        for index in secondary_data_df_nonan.loc[list_counts_start[c] : list_index_peak[p]].index:
            if len(list_index) == 1:
                h1 = secondary_data_df_nonan['ams' + str(list_index[0])].loc[index]
                h_ref = list_weights[0]*h1
            elif len(list_index) == 2:
                h1 = secondary_data_df_nonan['ams' + str(list_index[0])].loc[index]
                h2 = secondary_data_df_nonan['ams' + str(list_index[1])].loc[index]
                h_ref = list_weights[0]*h1 + list_weights[1]*h2
            elif len(list_index) == 3:
                h1 = secondary_data_df_nonan['ams' + str(list_index[0])].loc[index]
                h2 = secondary_data_df_nonan['ams' + str(list_index[1])].loc[index]
                h3 = secondary_data_df_nonan['ams' + str(list_index[2])].loc[index]
                h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3
            elif len(list_index) == 4:
                h1 = secondary_data_df_nonan['ams' + str(list_index[0])].loc[index]
                h2 = secondary_data_df_nonan['ams' + str(list_index[1])].loc[index]
                h3 = secondary_data_df_nonan['ams' + str(list_index[2])].loc[index]
                h4 = secondary_data_df_nonan['ams' + str(list_index[3])].loc[index]
                h_ref = list_weights[0]*h1 + list_weights[1]*h2 + list_weights[2]*h3 + list_weights[3]*h4

            df_reference_values.loc[index] = h_ref

    return df_reference_values

def coordinates_all_stations_in_range(loc_prim, loc_sec, y, station, ref1, ref2, ref3, ref4, delta, geo, ref_df, all_stations_in_range):
    
    if y == 'primary':
        coords_lon = loc_prim['lon']
        coords_lat = loc_prim['lat']
    elif y == 'secondary':
        coords_lon = loc_sec['lon']
        coords_lat = loc_sec['lat']
    elif y == 'both':
        coords_lon_prim = loc_prim['lon']
        coords_lat_prim = loc_prim['lat']
        coords_lon_sec = loc_sec['lon']
        coords_lat_sec = loc_sec['lat']

    if y == 'both':
        name_plot = 'Coordinates ' + y + ' networks'
        plt.scatter(x=coords_lon_prim, y=coords_lat_prim, s=20, color='red', label='primary network', marker='x', linewidth=1)
        plt.scatter(x=coords_lon_sec, y=coords_lat_sec, s=2, color='blue', label='secondary network', alpha=0.5)
        if type(station) == int:
            plt.scatter(loc_prim['lon'].loc[station], loc_prim['lat'].loc[station], color='black')
        plt.legend()
    else:
        name_plot = 'Coordinates ' + y + ' network: ams' + str(station)
        plt.scatter(x=coords_lon, y=coords_lat, s=10)
        if type(station) == int:
            if y == 'primary':
                plt.scatter(loc_prim['lon'].loc[station], loc_prim['lat'].loc[station], color='red')
            elif y == 'secondary':
                plt.scatter(loc_sec['lon'].loc[station], loc_sec['lat'].loc[station], color='red')

            if all_stations_in_range:
                for station_ref in ref_df.index:
                    plt.scatter(loc_sec['lon'].loc[station_ref], loc_sec['lat'].loc[station_ref], color='lime', s=10)
            else:
                try:
                    plt.scatter(loc_sec['lon'].loc[ref1], loc_sec['lat'].loc[ref1], color='lime', s=10)
                    plt.scatter(loc_sec['lon'].loc[ref2], loc_sec['lat'].loc[ref2], color='lime', s=10)
                    plt.scatter(loc_sec['lon'].loc[ref3], loc_sec['lat'].loc[ref3], color='lime', s=10)
                    plt.scatter(loc_sec['lon'].loc[ref4], loc_sec['lat'].loc[ref4], color='lime', s=10)
                except:
                    pass

    if geo == 'circle':
        kreis = Circle((loc_sec['lon'].loc[station], loc_sec['lat'].loc[station]), radius=1500, color='black', linewidth=0.5, fill=False)
        plt.gca().add_patch(kreis)
    elif geo == 'rectangle':
        quadrat = Rectangle((loc_sec['lon'].loc[station] - 1500, loc_sec['lat'].loc[station] - 1500), 3000, 3000, color='black', linewidth=0.5, fill=False)
        plt.hlines(loc_sec['lat'].loc[station], loc_sec['lon'].loc[station] - 1500, loc_sec['lon'].loc[station] + 1500, linewidths=0.5, color='black')
        plt.vlines(loc_sec['lon'].loc[station], loc_sec['lat'].loc[station] - 1500, loc_sec['lat'].loc[station] + 1500, linewidths=0.5, color='black')
        plt.gca().add_patch(quadrat) 

    plt.axis('equal')

    if type(delta) == int:
        try:
            plt.xlim(loc_sec['lon'].loc[station] - delta, loc_sec['lon'].loc[station] + delta)
        except:
            if ((loc_sec['lon'].loc[station] - delta) < loc_sec['lon'].min()):
                plt.xlim(loc_sec['lon'].min(), loc_sec['lon'].loc[station] + delta)
            if ((loc_sec['lon'].loc[station] + delta) > loc_sec['lon'].max()):
                plt.xlim(loc_sec['lon'].loc[station] - delta, loc_sec['lon'].max())
        try:
            plt.ylim(loc_sec['lat'].loc[station] - delta, loc_sec['lat'].loc[station] + delta)
        except:
            if ((loc_sec['lat'].loc[station] - delta) < loc_sec['lat'].min()):
                plt.ylim(loc_sec['lat'].min(), loc_sec['lat'].loc[station] + delta)
            if ((loc_sec['lat'].loc[station] + delta) > loc_sec['lat'].max()):
                plt.ylim(loc_sec['lat'].loc[station] - delta, loc_sec['lat'].max())
    else:
        pass
    
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(name_plot)

    plt.show()
    plt.close()
    
    return

def find_all_stations_in_range(loc_prim, loc_sec, station, delta, geo, all_stations_in_range):
    coordinates = loc_sec
    # finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    # set frame for search
    radius = 1500

    # set coordinates of the station
    lon_station = coordinates['lon'].loc[station]
    lat_station = coordinates['lat'].loc[station]

    list_reference_stations_lon = []
    list_reference_stations_lat = []
    list_station = []
    list_distance = []

    # find the 4 nearest stations in frame
    for i in coordinates.index:
        lon = coordinates['lon'].loc[i]
        lat = coordinates['lat'].loc[i]
        if (np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2) <= radius):
            if lon == lon_station and lat == lat_station:
                pass
            else:
                # print('lon:', lon, 'lat:', lat, '\nstation nr.:', i)
                # print('\n')

                distance = round(np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2), 2)
                
                list_distance.append(distance)
                list_station.append(i)
                list_reference_stations_lon.append(round(lon, 2))
                list_reference_stations_lat.append(round(lat, 2))
        
        array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
        df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

    if all_stations_in_range:
        coordinates_all_stations_in_range(loc_prim, loc_sec, 'secondary', station, '-', '-', '-', '-', delta, geo, df_reference_stations, all_stations_in_range)
    else:
        if len(df_reference_stations) == 0:
            pass
        elif len(df_reference_stations) == 1:
            coordinates_all_stations_in_range(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], '-', '-', '-', delta, geo, df_reference_stations, all_stations_in_range)
        elif len(df_reference_stations) == 2:
            coordinates_all_stations_in_range(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], df_reference_stations.index[1], '-', '-', delta, geo, df_reference_stations, all_stations_in_range)
        elif len(df_reference_stations) == 3:
            coordinates_all_stations_in_range(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], df_reference_stations.index[1], df_reference_stations.index[2], '-', delta, geo, df_reference_stations, all_stations_in_range)
        elif len(df_reference_stations) == 4:
            coordinates_all_stations_in_range(loc_prim, loc_sec, 'secondary', station, df_reference_stations.index[0], df_reference_stations.index[1], df_reference_stations.index[2], df_reference_stations.index[3], delta, geo, df_reference_stations, all_stations_in_range)

    return df_reference_stations

