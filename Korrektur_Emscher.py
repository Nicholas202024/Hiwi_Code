# import packages
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime
import functions_hiwi as fct
import Korrektur_Emscher as ke
import warnings
import xarray as xr
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle


###
# 
# 
# 
# 
# Funktionen die zur Korrektur der Emscher Daten allein mit PWS benötigt werden 
# 
# 
# 
# 
###

def list_nan_sequences_schnell(data, station):
    
    '''starts, ends, len_seq'''
    
    timedelta = datetime.timedelta(minutes=5)

    is_nan = data[station].isna() # gibt true zurück, wenn Wert NaN ist
    diff = is_nan.diff() # gibt true zurück, wenn Wert zu Nan oder Nan zu Wert springt

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
    
    return starts, ends, len_seq

def get_data_nan_seq_before_peak_new(data, station, quantile):

    # get info about nan sequences and peaks
    starts, ends, len_seq = ke.list_nan_sequences_schnell(data, station) # gives start, end and length of nan sequences
    peaks = data[station][data[station] > data[station].quantile(quantile)] # gives values + index of peak

    # check wich sequence has peak
    ends_plus_timedelta = ends + datetime.timedelta(minutes=5) # add timedelta to ends, because the peak is in the next time step
    peaks_mit_nan_seq = ends_plus_timedelta.intersection(peaks.index)

    # create mask to filter for starts of nan sequences with peaks
    mask = ends_plus_timedelta.isin(peaks_mit_nan_seq) # are the values of ends_plus_timedelta in ends_nan_seq_mit_peak, to get the place of starts of nan sequences with peaks
    starts_nan_seq_mit_peak = starts[mask]

    return starts_nan_seq_mit_peak, peaks_mit_nan_seq  

def coordinates_all_stations_in_range(loc_prim, loc_sec, y, station, frame, geo, radius, ref_df):
    
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

    if y == 'both': # plot both networks
        name_plot = 'Coordinates ' + y + ' networks'
        plt.scatter(x=coords_lon_prim, y=coords_lat_prim, s=20, color='red', label='primary network', marker='x', linewidth=1)
        plt.scatter(x=coords_lon_sec, y=coords_lat_sec, s=2, color='blue', label='secondary network', alpha=0.5)
        plt.legend()
    else: # plot one network, primary or secondary, with selected station (primary or secondary)
        name_plot = 'Coordinates ' + y + ' network: Station' + str(station)
        plt.scatter(x=coords_lon, y=coords_lat, s=10)
        if type(station) == int:
            if y == 'primary':
                plt.scatter(loc_prim['lon'].loc[station], loc_prim['lat'].loc[station], color='red')
            elif y == 'secondary':
                plt.scatter(loc_sec['lon'].loc[station], loc_sec['lat'].loc[station], color='red')

                # plot secondary reference stations of selected station
                for station_ref in ref_df.index:
                    plt.scatter(loc_sec['lon'].loc[station_ref], loc_sec['lat'].loc[station_ref], color='lime', s=10)

    # plot circle
    if geo == 'circle':
        kreis = Circle((loc_sec['lon'].loc[station], loc_sec['lat'].loc[station]), radius=radius, color='black', linewidth=0.5, fill=False)
        plt.gca().add_patch(kreis)
    # plot rectangle
    elif geo == 'rectangle':
        quadrat = Rectangle((loc_sec['lon'].loc[station] - radius, loc_sec['lat'].loc[station] - radius), radius*2, radius*2, color='black', linewidth=0.5, fill=False)
        plt.hlines(loc_sec['lat'].loc[station], loc_sec['lon'].loc[station] - radius, loc_sec['lon'].loc[station] + radius, linewidths=0.5, color='black')
        plt.vlines(loc_sec['lon'].loc[station], loc_sec['lat'].loc[station] - radius, loc_sec['lat'].loc[station] + radius, linewidths=0.5, color='black')
        plt.gca().add_patch(quadrat) 

    plt.axis('equal')

    # set frame of view for plot
    if type(frame) == int:
        try:
            plt.xlim(loc_sec['lon'].loc[station] - frame, loc_sec['lon'].loc[station] + frame)
        except:
            if ((loc_sec['lon'].loc[station] - frame) < loc_sec['lon'].min()):
                plt.xlim(loc_sec['lon'].min(), loc_sec['lon'].loc[station] + frame)
            if ((loc_sec['lon'].loc[station] + frame) > loc_sec['lon'].max()):
                plt.xlim(loc_sec['lon'].loc[station] - frame, loc_sec['lon'].max())
        try:
            plt.ylim(loc_sec['lat'].loc[station] - frame, loc_sec['lat'].loc[station] + frame)
        except:
            if ((loc_sec['lat'].loc[station] - frame) < loc_sec['lat'].min()):
                plt.ylim(loc_sec['lat'].min(), loc_sec['lat'].loc[station] + frame)
            if ((loc_sec['lat'].loc[station] + frame) > loc_sec['lat'].max()):
                plt.ylim(loc_sec['lat'].loc[station] - frame, loc_sec['lat'].max())
    else:
        pass
    
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(name_plot)

    plt.show()
    plt.close()
    
    return

def find_all_stations_in_range(loc_prim, loc_sec, station, frame, radius, plot=True):
    coordinates = loc_sec
    # finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen
    # set frame for search

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

    if plot:
        ke.coordinates_all_stations_in_range(loc_prim, loc_sec, 'secondary', station, frame, 'circle', radius, df_reference_stations)
       
    return df_reference_stations

def berechnung_gewichte_emscher(df_reference_stations):

    list_gewichte = []

    # berechnung Teiler
    teiler = 0
    for ref_station in df_reference_stations.index:
        teiler += (1/(df_reference_stations['distance'].loc[ref_station])**2)

    # berechnung Gewichte der Referenzstationen
    for ref_station in df_reference_stations.index:
        weight = (1/(df_reference_stations['distance'].loc[ref_station])**2)/teiler
        list_gewichte.append(weight)

    df_reference_stations['weight'] = list_gewichte

    return df_reference_stations

def berechnung_referenzniederschlag_emscher(df_reference_stations, data, data_nonan, station_zahl, starts_nan_seq_mit_peak, peaks_mit_nan_seq):
    
    # df zum speichern der Referenzniederschläge erstellen
    df_reference_values = data[[station_zahl]].copy()

    # abrufen der Informationen über NaN-Sequenz und Peaks
    # starts_nan_seq_mit_peak, peaks_mit_nan_seq = get_data_nan_seq_before_peak_new(data, station_zahl, 0.99)

    # berechnung Referenzniederschlag
    # für jede NaN-Sequenz und Peak
    for c, p in zip(range(0, len(starts_nan_seq_mit_peak)), range(0, len(peaks_mit_nan_seq))):
        # für jeden Index in der NaN-Sequenz
        for index in data_nonan.loc[starts_nan_seq_mit_peak[c] : peaks_mit_nan_seq[p]].index:
            h_ref = 0
            # für jede Referenzstation
            for ref_station in df_reference_stations.index:
                h_ref += data_nonan[ref_station].loc[index] * df_reference_stations['weight'].loc[ref_station]

            # speichern des Referenzniederschlags für den Index
            df_reference_values.loc[index] = h_ref

        # alle Werte außerhalb der NaN-Sequenz und Peaks auf NaN setzen
        if c == 0 and p == 0:
            df_reference_values.loc[(df_reference_values.index[0]) : (starts_nan_seq_mit_peak[c] - datetime.timedelta(minutes=5))] = np.nan
        if (c > 0 and p > 0) and ((c < len(starts_nan_seq_mit_peak) - 1) and (p < len(peaks_mit_nan_seq) - 1)):
            df_reference_values.loc[(peaks_mit_nan_seq[p - 1] + datetime.timedelta(minutes=5)) : (starts_nan_seq_mit_peak[c] - datetime.timedelta(minutes=5))] = np.nan
        if (c == len(starts_nan_seq_mit_peak) - 1) and (p == len(peaks_mit_nan_seq) - 1):
            df_reference_values.loc[(peaks_mit_nan_seq[p] + datetime.timedelta(minutes=5)) : (df_reference_values.index[-1])] = np.nan
    
    return df_reference_values

def correct_data_new_emscher(data, reference_df, station_zahl, starts_nan_seq_mit_peak, peaks_mit_nan_seq, correct_peak=True, correct_1_2=True, correct_0_pres_ref=True):

    # nans vor peaks korrigieren
    if correct_peak:

        # starts_nan_seq_mit_peak, peaks_mit_nan_seq = get_data_nan_seq_before_peak_new(data, station_zahl, quantile)
    
        data_corrected = data[[station_zahl]].copy() # copy the data to a new dataframe

        frequency = '5min'

        for i in range(len(peaks_mit_nan_seq)):

            datetime_index = pd.date_range(start=starts_nan_seq_mit_peak[i], end=peaks_mit_nan_seq[i], freq=frequency) # create a datetime index for the time period of the nan sequence before the peak
            sum = reference_df[station_zahl].loc[starts_nan_seq_mit_peak[i] : peaks_mit_nan_seq[i]].sum() # sum of the reference values for the time period of the nan sequence before the peak
            value_peak = data[station_zahl].loc[peaks_mit_nan_seq[i]] # value of the peak

            for index in datetime_index:
                try:
                    peak_portion = round(((reference_df[station_zahl].loc[index] / sum) * value_peak), 2)
                except ZeroDivisionError:
                    peak_portion = 0
                    
                data_corrected[station_zahl].loc[index] = peak_portion # replace the nan values with the calculated peak portion
            
    # 1er und 2er nan sequenzen korrigieren
    if correct_1_2:

        data_corrected = fct.einer_zweier_sequ_korrigieren(data_corrected, station_zahl, True, True)

    # nan sequenzen korrigieren, die mit 0 anfnagen und enden und bei denen die summe des niederschlag im referenz df 0 ist
    if correct_0_pres_ref:

        starts, ends, nan_sequs = fct.list_nan_sequences_schnell(data_corrected, station_zahl, frequency)

        list = []

        # erste und letzte nan sequenz extra kontrollieren, ob ganz am Anfang oder ganz am Ende von Dataframe
        if starts[0] == data_corrected.index[0] and ((data_corrected.loc[ends[0] + datetime.timedelta(minutes=5)]) == 0).bool():
            sum = reference_df[station_zahl].loc[starts[0] : ends[0]].values.sum()
            if sum == 0:
                list.append(0)

        if ends[-1] == data_corrected.index[-1] and ((data_corrected.loc[starts[-1] - datetime.timedelta(minutes=5)]) == 0).bool():
            sum = reference_df[station_zahl].loc[starts[-1] : ends[-1]].values.sum()
            if sum == 0:
                list.append(-1)

        for i in range(len(starts)):
            try:
                if ((data_corrected.loc[starts[i] - datetime.timedelta(minutes=5)]) == 0).bool() and ((data_corrected.loc[ends[i] + datetime.timedelta(minutes=5)]) == 0).bool():
                    sum = reference_df[station_zahl].loc[starts[i] : ends[i]].values.sum()
                    if sum == 0:
                        list.append(i)
            except KeyError:
                continue
            
        for i in list:
            data_corrected.loc[starts[i] : ends[i]] = 0

    return data_corrected

def correction_emscher_complete(loc_prim, loc_sec, data, data_nonan, station_zahl, frame, radius, quantile, plot=False, correct_peak=True, correct_1_2=True, correct_0_pres_ref=True):
    df_reference_stations = ke.find_all_stations_in_range(loc_prim, loc_sec, station_zahl, frame, radius, plot)
    if df_reference_stations.empty:
        print('No reference stations found within radius')
        return
    df_reference_stations = ke.berechnung_gewichte_emscher(df_reference_stations)
    starts_nan_seq_mit_peak, peaks_mit_nan_seq = ke.get_data_nan_seq_before_peak_new(data, station_zahl, quantile)
    df_reference_values = ke.berechnung_referenzniederschlag_emscher(df_reference_stations, data, data_nonan, station_zahl, starts_nan_seq_mit_peak, peaks_mit_nan_seq)
    data_corrected = ke.correct_data_new_emscher(data, df_reference_values, station_zahl, starts_nan_seq_mit_peak, peaks_mit_nan_seq, correct_peak=True, correct_1_2=True, correct_0_pres_ref=True)

    return data_corrected

###
# 
# 
# 
# 
# Funktionen die zur Korrektur der Emscher Daten mit PWS oder Pluvio benötigt werden 
# 
# 
# 
# 
###

def coordinates_all_stations_in_range_mit_primary(loc_prim, loc_sec, type_station, station, frame, geo, radius, ref_df, primary_found):

    # name plot
    name_plot = 'Plot Primär-/Sekundärnetzwerk: ' + type_station + 'station Nr. ' + str(station)

    # beide Netzwerke plotten
    plt.scatter(x=loc_sec['lon'], y=loc_sec['lat'], s=10, label='PWS')
    plt.scatter(x=loc_prim['lon'], y=loc_prim['lat'], s=30, color='black', marker='x', label='Primärstationen')

    # ausgewählte Station plotten
    if type(station) == int:
        if type_station == 'Primär':
            plt.scatter(loc_prim['lon'].loc[station], loc_prim['lat'].loc[station], color='red', label='Ausgewählte Station')
        elif type_station == 'Sekundär':
            plt.scatter(loc_sec['lon'].loc[station], loc_sec['lat'].loc[station], color='red', label='Ausgewählte Station')
            
            # Referenzstationen plotten
            if primary_found:
                plt.scatter(loc_prim['lon'].loc[ref_df.index[0]], loc_prim['lat'].loc[ref_df.index[0]], color='lime', s=30, marker='x', label='Primärreferenzstatoin(-en)')
                for station_ref in ref_df.index[1:]:
                    plt.scatter(loc_prim['lon'].loc[station_ref], loc_prim['lat'].loc[station_ref], color='lime', s=30, marker='x')
            else:
                plt.scatter(loc_sec['lon'].loc[ref_df.index[0]], loc_sec['lat'].loc[ref_df.index[0]], color='lime', s=10, label='PWS-Referenzstation(-en)')
                for station_ref in ref_df.index[1:]:
                    plt.scatter(loc_sec['lon'].loc[station_ref], loc_sec['lat'].loc[station_ref], color='lime', s=10)

    # plotte Kreis
    if geo == 'circle':
        kreis = Circle((loc_sec['lon'].loc[station], loc_sec['lat'].loc[station]), radius=radius, color='black', linewidth=0.5, fill=False)
        plt.gca().add_patch(kreis)

    plt.axis('equal')

    # Anzeigebereich festlegen
    if type(frame) == int:
        try:
            plt.xlim(loc_sec['lon'].loc[station] - frame, loc_sec['lon'].loc[station] + frame)
        except:
            if ((loc_sec['lon'].loc[station] - frame) < loc_sec['lon'].min()):
                plt.xlim(loc_sec['lon'].min(), loc_sec['lon'].loc[station] + frame)
            if ((loc_sec['lon'].loc[station] + frame) > loc_sec['lon'].max()):
                plt.xlim(loc_sec['lon'].loc[station] - frame, loc_sec['lon'].max())
        try:
            plt.ylim(loc_sec['lat'].loc[station] - frame, loc_sec['lat'].loc[station] + frame)
        except:
            if ((loc_sec['lat'].loc[station] - frame) < loc_sec['lat'].min()):
                plt.ylim(loc_sec['lat'].min(), loc_sec['lat'].loc[station] + frame)
            if ((loc_sec['lat'].loc[station] + frame) > loc_sec['lat'].max()):
                plt.ylim(loc_sec['lat'].loc[station] - frame, loc_sec['lat'].max())
    else:
        pass
    
    plt.legend()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(name_plot)

    plt.show()
    plt.close()
    
    return

def find_all_stations_in_range_mit_primary(loc_prim, loc_sec, station, frame, radius, plot=True):

    # finde stationen, die innerhalb eines bestimmten Bereich um die Station liegen

    # koordinaten der station
    lon_station = loc_sec['lon'].loc[station]
    lat_station = loc_sec['lat'].loc[station]

    list_reference_stations_lon = []
    list_reference_stations_lat = []
    list_station = []
    list_distance = []

    # check, ob Primärreferenz innerhalb Radius
    primary_found = False

    for i in loc_prim.index:
        lon = loc_prim['lon'].loc[i]
        lat = loc_prim['lat'].loc[i]
        if (np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2) <= radius):

            distance = round(np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2), 2)
                    
            list_distance.append(distance)
            list_station.append(i)
            list_reference_stations_lon.append(round(lon, 2))
            list_reference_stations_lat.append(round(lat, 2))

    if list_station:
        primary_found = True

    # finde die nächsten PWS, wenn keine Primärreferenz gefunden wurde
    if primary_found:
        pass
    else:
        for i in loc_sec.index:
            lon = loc_sec['lon'].loc[i]
            lat = loc_sec['lat'].loc[i]
            if (np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2) <= radius):
                if lon == lon_station and lat == lat_station:
                    pass
                else:
                    distance = round(np.sqrt((lon - lon_station)**2 + (lat - lat_station)**2), 2)
                    
                    list_distance.append(distance)
                    list_station.append(i)
                    list_reference_stations_lon.append(round(lon, 2))
                    list_reference_stations_lat.append(round(lat, 2))
        
    array_reference_stations = np.array([list_reference_stations_lon, list_reference_stations_lat, list_distance]).T
    df_reference_stations = pd.DataFrame(array_reference_stations, columns=['lon', 'lat', 'distance'], index=list_station)

    if plot:
        ke.coordinates_all_stations_in_range_mit_primary(loc_prim, loc_sec, 'Sekundär', station, frame, 'circle', radius, df_reference_stations, primary_found)
       
    return df_reference_stations, primary_found

def berechnung_referenzniederschlag_emscher_mit_primary(df_reference_stations, data_sec, data_prim_nonan, data_sec_nonan, station_zahl, starts_nan_seq_mit_peak, peaks_mit_nan_seq, primary_found):
    
    # df zum speichern der Referenzniederschläge erstellen
    df_reference_values = data_sec[[station_zahl]].copy()

    if primary_found:
        ref_station_values = data_prim_nonan
    else:
        ref_station_values = data_sec_nonan

    # abrufen der Informationen über NaN-Sequenz und Peaks
    # starts_nan_seq_mit_peak, peaks_mit_nan_seq = get_data_nan_seq_before_peak_new(data, station_zahl, 0.99)

    # berechnung Referenzniederschlag
    # für jede NaN-Sequenz und Peak
    for c, p in zip(range(0, len(starts_nan_seq_mit_peak)), range(0, len(peaks_mit_nan_seq))):
        # für jeden Index in der NaN-Sequenz
        for index in ref_station_values.loc[starts_nan_seq_mit_peak[c] : peaks_mit_nan_seq[p]].index:
            h_ref = 0
            # für jede Referenzstation
            for ref_station in df_reference_stations.index:
                h_ref += ref_station_values[ref_station].loc[index] * df_reference_stations['weight'].loc[ref_station]

            # speichern des Referenzniederschlags für den Index
            df_reference_values.loc[index] = h_ref

        # alle Werte außerhalb der NaN-Sequenz und Peaks auf NaN setzen
        if c == 0 and p == 0:
            df_reference_values.loc[(df_reference_values.index[0]) : (starts_nan_seq_mit_peak[c] - datetime.timedelta(minutes=5))] = np.nan
        if (c > 0 and p > 0) and ((c < len(starts_nan_seq_mit_peak) - 1) and (p < len(peaks_mit_nan_seq) - 1)):
            df_reference_values.loc[(peaks_mit_nan_seq[p - 1] + datetime.timedelta(minutes=5)) : (starts_nan_seq_mit_peak[c] - datetime.timedelta(minutes=5))] = np.nan
        if (c == len(starts_nan_seq_mit_peak) - 1) and (p == len(peaks_mit_nan_seq) - 1):
            df_reference_values.loc[(peaks_mit_nan_seq[p] + datetime.timedelta(minutes=5)) : (df_reference_values.index[-1])] = np.nan
    
    return df_reference_values

def correction_emscher_complete_mit_primary(loc_prim, loc_sec, data_sec, data_prim_nonan, data_sec_nonan, station_zahl, frame, radius, quantile, plot=False, correct_peak=True, correct_1_2=True, correct_0_pres_ref=True):
    df_reference_stations, primary_found = ke.find_all_stations_in_range_mit_primary(loc_prim, loc_sec, station_zahl, frame, radius, plot)
    if df_reference_stations.empty:
        print('No reference stations found within radius')
        return
    df_reference_stations = ke.berechnung_gewichte_emscher(df_reference_stations)
    starts_nan_seq_mit_peak, peaks_mit_nan_seq = ke.get_data_nan_seq_before_peak_new(data_sec, station_zahl, quantile)
    df_reference_values = ke.berechnung_referenzniederschlag_emscher_mit_primary(df_reference_stations, data_sec, data_prim_nonan, data_sec_nonan, station_zahl, starts_nan_seq_mit_peak, peaks_mit_nan_seq, primary_found)
    data_corrected = ke.correct_data_new_emscher(data_sec, df_reference_values, station_zahl, starts_nan_seq_mit_peak, peaks_mit_nan_seq, correct_peak, correct_1_2, correct_0_pres_ref)

    return data_corrected

###
# 
# 
# 
# 
# Funktionen die zur NaN-Analyse der Emscher Daten benötigt werden 
# 
# 
# 
# 
###

def get_statistics_emscher(data_uncorrected, data_corrected):

    list_nans_gesamt = []
    list_nan_sequences = []
    list_nan_sequences_1_2 = []
    list_nan_sequences_1_2_corr = []
    list_peaks_u = []
    list_peaks = []
    list_nans_gesamt_corr = []
    list_nan_sequences_corr = []
    list_verhaeltnis_nans = []
    list_verhaeltnis_nan_sequences = []

    for station in data_corrected.columns:
        sum_nan_u = data_uncorrected[station].isna().sum()
        list_u = ke.list_nan_sequences_schnell(data_uncorrected, station, '5min')[2]
        x_u, y_u = np.unique(list_u, return_counts=True)
        peaks_u = ke.get_data_nan_seq_before_peak_new(data_uncorrected, station, 0.99)[1]

        # print(station, 'unkorrigiert')
        # print('   nans gesamt: ', sum_nan_u)
        # print('   nan sequenzen: ', y_u.sum())
        # print('   davon 1er und 2er nan sequenzen: ', y_u[0:2].sum())
        # print('   peaks: ', len(peaks_u))

        list_nans_gesamt.append(sum_nan_u)
        list_nan_sequences.append(y_u.sum())
        list_nan_sequences_1_2.append(y_u[0:2].sum())
        list_peaks_u.append(len(peaks_u))

        sum_nan = data_corrected[station].isna().sum()
        list = ke.list_nan_sequences_schnell(data_corrected, station, '5min')[2]
        x, y = np.unique(list, return_counts=True)
        peaks = ke.get_data_nan_seq_before_peak_new(data_corrected, station, 0.99)[1]

        # print(station, 'korrigiert')
        # print('   nans gesamt: ', sum_nan)
        # print('   nan sequenzen: ', y.sum())
        # print('Verhältnis korrigiert zu unkorrigiert: ', round(((sum_nan_u - sum_nan)/sum_nan_u)*100, 2), '% werden korrigiert')

        list_nans_gesamt_corr.append(sum_nan)
        list_nan_sequences_corr.append(y.sum())
        list_verhaeltnis_nans.append(round(((sum_nan_u - sum_nan)/sum_nan_u)*100))
        list_verhaeltnis_nan_sequences.append(round(((y_u.sum() - y.sum())/y_u.sum())*100))
        list_peaks.append(len(peaks))
        list_nan_sequences_1_2_corr.append(y[0:2].sum())

    df_emscher_statistik = pd.DataFrame(index=['NaNs gesamt', 'NaN-Sequenzen', '1er und 2er NaN-Sequenzen', 'Peaks', '--------------------------------------------------', 'NaNs nach Korrektur', 'NaN-Sequenzen nach Korrektur', '1er und 2er NaN-Sequenzen nach Korrektur', 'Peaks nach Korrektur', '--------------------------------------------------', '% NaNs korrigiert', '% NaN-Sequenzen korrigiert'], data=[list_nans_gesamt, list_nan_sequences, list_nan_sequences_1_2, list_peaks_u, [], list_nans_gesamt_corr, list_nan_sequences_corr, list_nan_sequences_1_2_corr, list_peaks, [], list_verhaeltnis_nans, list_verhaeltnis_nan_sequences], columns=data_corrected.columns)
    df_emscher_statistik = df_emscher_statistik.fillna('')

    return df_emscher_statistik