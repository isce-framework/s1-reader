import argparse
import glob
from s1reader.s1_orbit import get_orbit_file_from_list
from s1reader.s1_reader import load_bursts
from osgeo import gdal, osr
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from shapely import wkt
import fiona

def command_line_parser():
    '''
    Command line parser
    '''
    parser = argparse.ArgumentParser(description="""
                                     Create a burst map for a single slc""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--slc', type=str, action='store',
                        dest='slc',
                        help="slc to map")
    parser.add_argument('-d', '--orbit-dir', type=str, dest='orbit_dir',
                        help="Directory containing orbit files")
    parser.add_argument('-x', '--x-spacing', type=float, default=5,
                        dest='x_spacing',
                        help='Spacing of the geogrid in x direction')
    parser.add_argument('-y', '--y-spacing', type=float, default=10,
                        dest='y_spacing',
                        help='Spacing of the geogrid in y direction')
    parser.add_argument('-e', '--epsg', type=int, dest='epsg',
                        help='EPSG for output coordinates')
    parser.add_argument('-o', '--output', type=str, default='burst_map.gpkg',
                        dest='output',
                        help='Output filename for burst map')
    return parser.parse_args()


def burst_map(slc, orbit_dir, x_spacing,
              y_spacing, epsg,
              output_filename):
    """
    Create a CSV of SLC metadata and plot bursts
    Parameters:
    slc: str
      Path to SLC file
    orbit_dir: str
      Path to directory containing orbit files
    x_spacing: float
      Spacing of the geogrid in the x direction
    y_spacing: float
      Spacing to the geogrid in the y direction
    epsg: int
      EPSG code for the output coodrdinates
    output_filename: str
      Filename used for the output CSV, shp, html, and kml files
    
    Returns:
    output_filename.csv, output_filename.shp, output_filename.html, output_filename.kml
    """
    
    # Initialize dictionary that will contain all the info for geocoding
    burst_map = {'burst_id':[], 'length': [], 'width': [],
                 'spacing_x': [], 'spacing_y':[], 'min_x': [],
                 'max_x': [], 'min_y': [], 'max_y': [], 'first_valid_line': [],
                 'last_valid_line':[], 'first_valid_sample':[], 'last_valid_sample':[],
                 'border':[]}
    i_subswath = [1, 2, 3]
    pol = 'vv'
    orbit_list = glob.glob(f'{orbit_dir}/*EOF')
    orbit_path = get_orbit_file_from_list(slc, orbit_list)

    for subswath in i_subswath:
        ref_bursts = load_bursts(slc, orbit_path, subswath, pol)
        for burst in ref_bursts:
            burst_map['burst_id'].append(burst.burst_id)
            burst_map['length'].append(burst.shape[0])
            burst_map['width'].append(burst.shape[1])
            burst_map['spacing_x'].append(x_spacing)
            burst_map['spacing_y'].append(y_spacing)
            burst_map['first_valid_line'].append(burst.first_valid_line)
            burst_map['last_valid_line'].append(burst.last_valid_line)
            burst_map['first_valid_sample'].append(burst.first_valid_sample)
            burst_map['last_valid_sample'].append(burst.last_valid_sample)

            poly = burst.border[0]
            # Give some margin to the polygon
            margin = 0.001
            poly = poly.buffer(margin)
            burst_map['border'].append(Polygon(poly.exterior.coords).wkt)

            # Transform coordinates from lat/long to EPSG
            llh = osr.SpatialReference()
            llh.ImportFromEPSG(4326)
            tgt = osr.SpatialReference()

            tgt_x, tgt_y = [], []
            x, y = poly.exterior.coords.xy
            tgt.ImportFromEPSG(int(epsg))
            trans = osr.CoordinateTransformation(llh, tgt)
            for lx, ly in zip(x, y):
                dummy_y, dummy_x, dummy_z = trans.TransformPoint(ly, lx, 0)
                tgt_x.append(dummy_x)
                tgt_y.append(dummy_y)
            
            if epsg == 4326:
              x_min = x_spacing * (min(tgt_x) / x_spacing)
              y_min = y_spacing * (min(tgt_y) / y_spacing)
              x_max = x_spacing * (max(tgt_x) / x_spacing)
              y_max = y_spacing * (max(tgt_y) / y_spacing)
            else:
              x_min = x_spacing * round(min(tgt_x) / x_spacing)
              y_min = y_spacing * round(min(tgt_y) / y_spacing)
              x_max = x_spacing * round(max(tgt_x) / x_spacing)
              y_max = y_spacing * round(max(tgt_y) / y_spacing) 

            # Allocate coordinates inside the dictionary
            burst_map['min_x'].append(x_min)
            burst_map['min_y'].append(y_min)
            burst_map['max_x'].append(x_max)
            burst_map['max_y'].append(y_max)

    # Save generated burst map as csv
    data = pd.DataFrame.from_dict(burst_map)
    data.to_csv(output_filename)

    # Create GeoDataFrame to plot bursts on a map
    df = data
    df['border'] = df['border'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326')
    gdf.rename(columns={'border': 'geometry', inplace=True}).set_geometry('geometry')
    
    # Save the GeoDataFrame as a shapefile (some people may prefer the format)
    gdf.to_file(f'{output_filename}.shp')
    
    # Save the GeoDataFrame as a kml
    fiona.supported_drivers['KML'] = 'rw'
    gdf.to_file(f'{output_filename}.kml', driver='KML')
    
    
    # Plot bursts on an interactive map
    m = gdf.explore(
        column="burst_id", # make choropleth based on "Burst ID" column
        tooltip="burst_id", # show "Burst ID" value in tooltip (on hover)
        popup=True, # show all values in popup (on click)
        tiles="CartoDB positron", # use "CartoDB positron" tiles
        cmap="Set1", # use "Set1" matplotlib colormap
        style_kwds=dict(color="black") # use black outline
       )

    m.save(f'{output_filename}.html')
    

if __name__ == '__main__':
    cmd = command_line_parser()
    burst_map(cmd.slc, cmd.orbit_dir,
              cmd.x_spacing, cmd.y_spacing,
              cmd.epsg, cmd.output)
