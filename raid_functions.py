import glob, os
import pandas as pd
import numpy as np
import seaborn as sns
import h5py
import scipy
import cv2
import statsmodels.api as sm
import statsmodels.tsa.stattools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#pull all tracks for one ant
def gettracks(ant, movie):
    movie = h5py.File(movie, 'r')

    if ant in list(movie.keys()):
        dset = movie[ant]
        xy = np.zeros(dset.shape, dtype = 'float')
        dset.read_direct(xy)
        return xy
    else:
        return np.nan

def getcolorlist(movie):
    movie = h5py.File(movie, 'r')
    return list(movie.keys())

def getframenums(colony, event, coldata):
    """Take colony name, event number, and coldata DF as inputs. Return
    five frame numbers which may include NaNs"""

    df = coldata[coldata['Colony'] == colony]
    row = df[df['Event'] == event]
    if row.empty:
        return np.NaN
    else:
        recstart = float(row['Recstart'].iloc[0])
        respstart = float(row['Respstart'].iloc[0])
        respstop = float(row['Respstop'].iloc[0])
        pullstart = float(row['Pullstart'].iloc[0])
        pullstop = float(row['Pullstop'].iloc[0])
        return recstart, respstart, respstop, pullstart, pullstop

def getnestcoordinates(colony, event, coldata):
    """Run to find nest centre, using colony name, event number, and coldata
        DF as input. Outputs nest coordinates and scale.
        DO NOT RUN IF SAVED COORDINATES ARE AVAILABLE"""

    colonydata = coldata[coldata['Colony'] == str(colony)]
    colonydata = colonydata[colonydata['Event'] == float(event)]
    hdd = colonydata['HDD'].iloc[0]
    cam = colonydata['Cam'].iloc[0]

    #locate arena file
    items = os.listdir(str(hdd))
    path = ''

    for name in items:
        if name.endswith(cam[-1:]):
            path = name

    arenafile = str(hdd) + '/' + str(path) + '/' + 'arena_mask.png'
    sc = str(hdd) + '/' + str(path) + '/' + 'scale.txt'
    sc = open(sc, 'r')
    scale = sc.read()
    sc.close()

    #find circles in arenafile
    img = cv2.imread(arenafile)
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,dp = 2, minDist = 500,
                           param1 = 300, param2 = 50,
                           minRadius= 100, maxRadius = 700)

    if str(type(circles)) == '<class \'numpy.ndarray\'>':
        circles = np.uint16(np.around(circles))
    else:
        return 'error'

    #find nest coordinates
    circ = pd.DataFrame(circles[0, :])
    circ.columns = ['x', 'y', 'r']
    circ = circ.sort_values('r', ascending = True)
    circ = circ.reset_index(drop = True)

    nest_x = float(circ['x'].iloc[0])
    nest_y = float(circ['y'].iloc[0])
    nest_centre = [nest_x, nest_y]

    return nest_centre, scale


def getuntaggedmovie(colony, event, coldata):
    """Takes colony, event, and coldata DF as input. Returns
    all path to movie, scale, and nest and food coordinates
    """

    colonydata = coldata[coldata['Colony'] == str(colony)]
    colonydata = colonydata[colonydata['Event'] == float(event)]
    hdd = str(colonydata['HDD'].iloc[0])
    cam = str(colonydata['CAM'].iloc[0])
    m = int(colonydata['First_video'].iloc[0])
    f = int(colonydata['Folder'].iloc[0]) - 1

    #scale = float(colonydata['Scale'].iloc[0])
    #nest = (colonydata['nest_x'].iloc[0], colonydata['nest_y'].iloc[0])
    #food = colonydata['Food'].iloc[0]

    #locate movie
    items = os.listdir(str(hdd))
    mov_items = []
    flist = []
    fpath = ''
    path = ''
    path2 = ''
    mov = ''

    #find cam folder
    for name in items:
        if name.endswith(cam[-1:]):
            path = name

    #if items produces list of folders that contains cam folder, select correct folder
    if path == '':
        for name in sorted(items):
            if name[0:3] == 'FOR':
                flist.append(name)
        fpath = flist[f]

        for name in os.listdir(str(hdd + '/' + fpath + '/')):
            if name.endswith(cam[-1:]):
                path = name

    slash = ''
    if fpath != '':
        slash = '/'

    #fullpath
    path2 = hdd + '/' + fpath + slash + path + '/'

    #list all movie files
    for name in os.listdir(path2):
        if '.mat' in str(name) and 'xy' in str(name):
            mov_items.append(name)

    #find correct movie
    for i in mov_items:
        j = i.split('_')
        if m >= float(j[1]) and m <= float(j[2].split('.')[0]):
            mov = i

    movie = hdd + '/' + fpath + slash + path + '/' + str(mov)

    if len(mov) == 0:
        return np.NaN
    else:
        return movie #, nest, scale, food


#measure instantaneous and cumulative distance travelled for each ant for an entire movie
#output pandas dataframe with x,y,framenum, instantaneous and cumulative distances travelled

def getdata(ant, colony, event, coldata):
    """take ant name, colony, event, and coldata DF as input
    and return DF with interesting features for the ant
    """

    movie, scale, nest, food = getmovie(colony, event, coldata)

    ##GET XY COORDINATES AND HEADING ANGLE FOR A SINGLE ANT IN A SINGLE MOVIE
    xy = gettracks(ant, movie)
    initdata = pd.DataFrame(xy.transpose())
    initdata.columns = ['x', 'y', 'orient']

    #Use rolling window to smooth data to 0.5sec resolution
    data = pd.DataFrame(initdata['x'].rolling(5,
        min_periods = 3, center = True).mean())
    data['y'] = pd.DataFrame(initdata['y'].rolling(5,
        min_periods = 3, center = True).mean())
    data['head_orient'] = pd.DataFrame((initdata['orient'].rolling(5, min_periods = 3,
        center = True).apply(lambda x: scipy.stats.circmean(x,
        high = np.pi, low = -np.pi), raw = False)))

    #store frame numbers
    data['frame'] = data.index

    #COMPUTATIONS
    #calculate instantaneous distance and direction of travel
    dx = (data['x'] - data['x'].shift())
    dy = (data['y'] - data['y'].shift())
    data['s_inst'] = np.sqrt(dx**2 + dy**2)
    data['vel_orient'] = np.arctan2(dy, dx)

    try:
        data['s_inst'].iloc[0] = 0
    except:
        return data

    #calculate instantaneous time
    data['t_inst'] = (data['frame'].diff())/10

    #DISTANCE TO NEST
    #GENERALISE IMAGE INPUT
    nest = ''.join(nest.split())
    nest = nest[1:-1]
    nest = nest.split(',')

    nest_x = float(nest[0]) * scale
    nest_y = float(nest[1]) * scale

    #calculate distance of ant to the nest in SI units
    data['distfromnest'] = np.sqrt((data['x'] - nest_x)**2 + (data['y'] - nest_y)**2)
    data['pos'] = 'in'

    #food position
    food = ''.join(food.split())
    food = food[1:-1]
    food = food.split(',')
    food_x = float(food[0])
    food_y = float(food[1])
    data['distfromfood'] = np.sqrt((data['x'] - food_x)**2 + (data['y'] - food_y)**2)

    radius = 0.018 #1.8cm is the distance from nest centre to tunnel entrance
    foodradius = 0.003 #ant is on food is she is within 3mm of food

    data.loc[data['distfromnest'] > radius, 'pos'] = 'out'
    data.loc[data['distfromfood'] < foodradius, 'pos'] = 'food'

    return data


#measure instantaneous and cumulative distance travelled for each ant for an entire movie
#output pandas dataframe with x,y,framenum, instantaneous and cumulative distances travelled

def getuntaggeddata(colony, event, coldata):
    """take ant name, colony, event, and coldata DF as input
    and return DF with interesting features for the ant
    """

    movie = getuntaggedmovie(colony, event, coldata)
    recstart, respstart, respstop, pullstart, pullstop = getframenums(colony, event, coldata)

    ##GET XY COORDINATES AND HEADING ANGLE FOR A SINGLE ANT IN A SINGLE MOVIE
    f1 = h5py.File(movie, 'r')

    data = pd.DataFrame()
    data['nants'] = f1['nants'][0]
    data['x'] = f1['xy'][0]
    data['y'] = f1['xy'][1]
    data['frame'] = f1['frame'][0]
    #data['orient'] = f1['orient'][0]
    data['area'] = f1['area'][0]

    #Use rolling window to smooth data to 0.5sec resolution
    #data['x'] = pd.DataFrame(data['x'].rolling(5,
    #    min_periods = 5, center = True).mean())
    #data['y'] = pd.DataFrame(data['y'].rolling(5,
    #    min_periods = 5, center = True).mean())

    #COMPUTATIONS
    #calculate instantaneous distance and direction of travel
    dx = (data['x'] - data['x'].shift())
    dy = (data['y'] - data['y'].shift())
    data['s_inst'] = np.sqrt(dx**2 + dy**2)
    data['vel_orient'] = np.arctan2(dy, dx)

    try:
        data['s_inst'].iloc[0] = 0
    except:
        return data

    #calculate instantaneous time
    data['t_inst'] = (data['frame'].diff())/10


    ##DISTANCE TO NEST

    #nest_x = nest[0]
    #nest_y = nest[1]

    #calculate distance of ant to the nest in SI units
    #data['distfromnest'] = np.sqrt((data['x'] - nest_x)**2 + (data['y'] - nest_y)**2)

    ##food position
    #food = ''.join(food.split())
    #food = food[1:-1]
    #food = food.split(',')
    #food_x = float(food[0])
    #food_y = float(food[1])
    #data['distfromfood'] = np.sqrt((data['x'] - food_x)**2 + (data['y'] - food_y)**2)

    #radius = 0.018 #1.8cm is the distance from nest centre to tunnel entrance
    #foodradius = 0.003 #ant is on food is she is within 3mm of food

    #data.loc[data['distfromnest'] > radius, 'pos'] = 'out'
    #data.loc[data['distfromfood'] < foodradius, 'pos'] = 'food'

    return data

def getexitmovie(colony, event, coldata):
    """Takes colony, event, and coldata DF as input. Returns
    all path to movie, scale, and nest and food coordinates
    """

    colonydata = coldata[coldata['Colony'] == str(colony)]
    colonydata = colonydata[colonydata['Event'] == float(event)]
    gs = str(colonydata['GS'].iloc[0])
    gs_start = str(colonydata['GS_start'].iloc[0])
    hdd = str(colonydata['HDD'].iloc[0])
    cam = str(colonydata['CAM'].iloc[0])
    m = int(colonydata['First_video'].iloc[0])
    f = int(colonydata['Folder'].iloc[0]) - 1

    #locate movie
    items = os.listdir(str(hdd))
    mov_items = []
    flist = []
    fpath = ''
    path = ''
    path2 = ''
    mov = ''

    #find cam folder
    for name in items:
        if name.endswith(cam[-1:]):
            path = name

    #if items produces list of folders that contains cam folder, select correct folder
    if path == '':
        for name in sorted(items):
            if name[0:3] == 'FOR':
                flist.append(name)
        fpath = flist[f]

        for name in os.listdir(str(hdd + '/' + fpath + '/')):
            if name.endswith(cam[-1:]):
                path = name

    slash = ''
    if fpath != '':
        slash = '/'

    #fullpath
    path2 = hdd + '/' + fpath + slash + path + '/'

    #list all movie files
    for name in os.listdir(path2):
        if '.mat' in str(name) and 'exit' in str(name):
            mov_items.append(name)

    #find correct movie
    for i in mov_items:
        j = i.split('_')
        if m >= float(j[2]) and m <= float(j[3].split('.')[0]):
            mov = i

    movie = hdd + '/' + fpath + slash + path + '/' + str(mov)

    if len(mov) == 0:
        return np.NaN
    #'SHIT SHIT SHIT'
    else:
        return movie, gs, gs_start

#measure instantaneous and cumulative distance travelled for each ant for an entire movie
#output pandas dataframe with x,y,framenum, instantaneous and cumulative distances travelled

def getexitdata(colony, event, coldata):
    """take ant name, colony, event, and coldata DF as input
    and return DF with interesting features for the ant
    """

    movie, gs, gs_start = getexitmovie(colony, event, coldata)

    ##GET XY COORDINATES AND HEADING ANGLE FOR A SINGLE ANT IN A SINGLE MOVIE
    exits = h5py.File(movie, 'r')
    exits = exits['exit_times'][:,0]

    df = pd.DataFrame()
    df['exits'] = exits
    df['normexits'] = (df['exits'] - df['exits'].min())
    df.reset_index()
    df['ind'] = df.index
    df['interval'] = df['exits'].diff()
    df['colony'] = colony
    df['event'] = event
    df['event_id'] = colony + str(event)
    df['gs'] = gs
    df['gs_start'] = gs_start

    return df

#calculate curvature of a path (from https://stackoverflow.com/a/28270382)
def getcurvature(a):
    #uses method of finite differences.
    #first, calculate derivatives of x and y
    a = a[[0,1],:]
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])
    velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])

    #instantaneous speed
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    #to find the unit tangent vector, transform ds_dt so it has the same size as the velocity vector
    tangent = np.array([1/ds_dt] * 2).transpose() * velocity

    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    #find derivatives of the tangent vector
    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])

    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)

    normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt

    #the normal vector represents the direction in which the curve is turning

    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5

    return curvature

def get_head_orientations(data, column):
    bins = np.linspace(-np.pi, np.pi, 72)
    labels = bins[:-1]
    temp = pd.DataFrame(pd.cut(data[column], bins = bins,
                    labels = labels).value_counts())
    temp = temp.reset_index()
    temp.columns = ['bin', 'count']
    return temp


def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    #from https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

#find frames at which ant is halfway to food, or halfway to nest
def partwaydistfind(df, key = 'distfromnest', frac = 0.5):
    """take dataframe of the form produced by getdata(), a string,
    and a float as input. String names the relevant distance:
    either distance from the nest (default) or distance from the food.
    Float names fraction of max distance to nest/food at which to
    calculate distance
    """
    halfway = np.round(df[key].max() * frac, decimals = 4)
    return df.loc[df[key].sub(halfway).abs().idxmin()]


def getphaseperframe(df):
    if len(df['phase'].unique()) < 1:
        raise ValueError('No elements in Series')
    else:
        return str(df.phase.value_counts().idxmax())


def filter_out_lines(x,y,dmin):
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    ix = np.where(~np.isnan(d) & (d>dmin))[0]
    if ix.size>0:
        x[ix+1]=np.nan
        y[ix+1]=np.nan
    return x,y
