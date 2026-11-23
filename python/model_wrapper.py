import os
from pathlib import Path
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
from hydromt_sfincs import SfincsModel, utils
from hydromt._utils import log
import matplotlib.pyplot as plt
from python.twl_boundary import create_twl_boundary
from python.shoreline_transformation import shoreline_translation, get_GCTR_data, subsidence_translation
from python.HGRAPHER_functions_mod import *
import yaml
import subprocess


def create_subgrid_catalog(model_root,da):
    crs = int(da.rio.crs.to_epsg())

    catalog = {
        "large_model_dep": {
            "data_type": "RasterDataset",
            "uri": "gis/dep_input.tif",
            "driver": {
                "name": "rasterio",
                "options": {
                    "chunks": {
                        "x": 1000,
                        "y": 1000,
                    }
                }
            },
            "metadata": {
                "category": "topography",
                "crs": crs,
                "nodata": -9999,
            },
            "data_adapter": {
                "rename": {
                    "large_model_dep": "elevtn"
                }
            }
        },

        "large_model_manning": {
            "data_type": "RasterDataset",
            "uri": "gis/manning_input.tif",
            "driver": {
                "name": "rasterio",
                "options": {
                    "chunks": {
                        "x": 1000,
                        "y": 1000,
                    }
                }
            },
            "metadata": {
                "category": "roughness",
                "crs": crs,
                "nodata": -9999,
            }
        }
    }

    shutil.copy2(Path(model_root) / "gis" / "dep.tif",
                Path(model_root) / "gis" / "dep_input.tif")

    shutil.copy2(Path(model_root) / "gis" / "manning.tif",
                Path(model_root) / "gis" / "manning_input.tif")

    catalog_subgrid = Path(model_root) / "subgrid.yml"

    with open(catalog_subgrid, "w") as f:
        yaml.safe_dump(catalog, f, sort_keys=False)

    return catalog_subgrid

def difference_map(dep_old,dep_new,lines=[],axis=0):
    diff = dep_new - dep_old

    plt.figure(figsize=(10,8))

    v = np.nanpercentile(np.abs(diff), 99)

    plt.pcolormesh(
        # da.xc,
        # da.yc,
        diff,
        shading="auto",
        cmap="RdBu_r",
        vmin=-v,
        vmax=v
    )

    # plt.plot(
    #     xy_samp[:,0],
    #     xy_samp[:,1],
    #     'k',
    #     lw=2,
    #     label='original shoreline'
    # )

    # plt.plot(
    #     xy_shift[:,0],
    #     xy_shift[:,1],
    #     'r--',
    #     lw=2,
    #     label='shifted shoreline'
    # )

    # n_lines = [100,350,700]
    if axis == 0:
        for line in lines:
            plt.axhline(
                line,
                color='red',
                ls='--',
                lw=2,
                # label='shifted shoreline'
            )
    else:
        for line in lines:
            plt.axvline(
                line,
                color='red',
                ls='--',
                lw=2,
                # label='shifted shoreline'
            )



    plt.colorbar(label="Elevation change (m)")
    plt.legend()
    # plt.axis("equal")
    plt.title("DEM change due to shoreline translation")
    plt.show()


def profile_check(dep_old,dep_new,res,n,axis=0,direction='left'):

    # n = 100
    if axis == 0:
        z0 = dep_old[n, :].astype(float)
        z1 = dep_new[n, :].astype(float)
    else:
        z0 = dep_old[:, n].astype(float)
        z1 = dep_new[:, n].astype(float)

    # mask nodata
    z0[z0 == -9999] = np.nan
    z1[z1 == -9999] = np.nan

    # x coordinate in meters
    x = np.arange(z0.size) * res

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(11, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # ------------------
    # profiles
    # ------------------
    ax1.plot(x, z0, 'k', lw=2, label='original profile')
    ax1.plot(x, z1, 'r', lw=2, label='modified profile')

    ax1.axhline(0, color='gray', ls='--', alpha=0.5)



    # shoreline location
    try:
        valid = np.isfinite(z0)
        shore0 = np.interp(0, z0[valid], x[valid])

        valid = np.isfinite(z1)
        shore1 = np.interp(0, z1[valid], x[valid])

        dy = shore1-shore0
        if direction == 'left':
            dy = dy
        elif direction =='right':
            dy = -dy

        ax1.axvline(shore0, color='k', ls=':')
        ax1.axvline(shore1, color='r', ls=':')

        ax1.annotate(f"shoreline shift = {dy:.2f} m",
                    xy=[shore0-50,0],
                    ha='right')


        print(f"shoreline shift = {dy:.2f} m")

    except:
        print("Could not find shoreline crossing")

    ax1.set_ylabel("Elevation (m)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ------------------
    # difference
    # ------------------
    diff = z1 - z0
    mask = np.isfinite(diff)
    ax2.plot(x, diff, color='blue')
    ax2.axhline(0, color='k', lw=1)
    # if direction == 'left':
    #     ax2.set_xlim(x[mask].min(), x[mask].min()+3000)
    # if direction == 'right':
    #     ax2.set_xlim(x[mask].max()-3000, x[mask].max())
    ax2.set_xlim((x[mask].min(),x[mask].max()))
    ax2.set_ylabel("Δz (m)")
    ax2.set_xlabel("Cross-shore distance (m)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def summarize_flood(da_hmax, scenario, dx, dy):

    cell_area = dx * dy
    print(dx,dy)
    n_valid = da_hmax.notnull().sum().compute().item()

    return {
        "scenario": scenario,
        "flooded_cells": n_valid,
        "flood_extent_m2": n_valid * cell_area,
        "flood_extent_km2": n_valid * cell_area / 1e6,
        "mean_depth_m": da_hmax.mean(skipna=True).compute().item(),
        "max_depth_m": da_hmax.max().compute().item(),
        "volume_m3": (
            da_hmax.sum(skipna=True).compute().item()
            * cell_area
        ),
    }

def post_process_output(model_root,yaml_path,mask_geojson=None,plot_image=False):
    
    mod = SfincsModel(model_root, data_libs=[yaml_path], mode="r")
    mod.output.read()
    # list(mod.output.data.keys())

    # read global surface water occurance (GSWO) data to mask permanent water
    gswo = mod.data_catalog.get_rasterdataset("gswo", geom=mod.region, buffer=10)


    hmin = 0.05  # minimum flood depth [m] to plot

    if (model_root / 'subgrid').exists():
        # first we are going to select our highest-resolution elevation dataset
        depfile = os.path.join(model_root, "subgrid", "dep_subgrid.tif")
        da_dep = mod.data_catalog.get_rasterdataset(depfile)
        
        # secondly we are reading in the model results
        da_zsmax = mod.output.data["zsmax"].max(dim="timemax")

        # Fourthly, we downscale the floodmap
        da_hmax = utils.downscale_floodmap(
            zsmax=da_zsmax,
            dep=da_dep,
            hmin=hmin,
            # gdf_mask=gdf,
            # floodmap_fn=os.path.join(model_root, "floodmap.tif") # uncomment to save to <mod.root>/floodmap.tif
        )
        gswo_mask = gswo.raster.reproject_like(da_dep, method="max") <= 5

        da_hmax = da_hmax.where(gswo_mask).where(da_hmax > hmin)

    else:
        # permanent water where water occurence > 5%
        gswo_mask = gswo.raster.reproject_like(mod.grid.data, method="max") <= 5

        # hmax is computed by SFINCS and read-in from the sfincs_map.nc file
        da_hmax = mod.output.data["hmax"].max(dim="timemax")

        # get overland flood depth with GSWO and set minimum flood depth
        da_hmax = da_hmax.where(gswo_mask).where(da_hmax > hmin)

    # update attributes for colorbar label later
    da_hmax.attrs.update(long_name="flood depth", unit="m")

    # -----------------------------
    # additional polygon mask
    # -----------------------------
    if mask_geojson is not None:
        gdf_mask = gpd.read_file(mask_geojson)

        # match CRS
        if gdf_mask.crs != da_hmax.raster.crs:
            gdf_mask = gdf_mask.to_crs(da_hmax.raster.crs)

        # True inside polygon
        polygon_mask = da_hmax.raster.geometry_mask(gdf_mask, all_touched=True)

        # keep only inside polygon
        da_hmax = da_hmax.where(polygon_mask)

    if plot_image:
        # create hmax plot and save to mod.root/figs/hmax.png
        fig, ax = mod.plot_basemap(
            fn_out=None,
            figsize=(8, 6),
            variable=da_hmax,
            plot_bounds=False,
            plot_geoms=False,
            # bmap="sat",
            zoomlevel=12,
            vmin=0,
            vmax=2.0,
            cmap=plt.cm.viridis,
            cbar_kwargs={"shrink": 0.6, "anchor": (0, 0)},
        )
        ax.set_title(f"SFINCS maximum water depth")

        return da_hmax, fig, ax
    
        # plt.savefig(join(mod.root, 'figs', 'hmax.png'), dpi=225, bbox_inches="tight")
    return da_hmax, mod

def run_simulation(src,model_root):
    ## run model
    # src = r"f:\TUD_CEG_Repository\MSc_Thesis\Code\run.bat"
    dst = Path(model_root) / 'run.bat'

    if not dst.exists():
        shutil.copy(src, dst)

    run_path = model_root
    # cur_dir = os.getcwd()

    # os.chdir(run_path)
    # os.system("run.bat")
    # os.chdir(cur_dir)

    subprocess.run(
        "run.bat",
        cwd=run_path,
        shell=True,
        check=True
    )

def run_model_scenario(base_model, scenario,
                       erosion = False, subsidence = False, nyear = 50,
                       region = None,
                       cl_file = 'gctr',
                       use_subgrid = False, nr_subgrid = 6,
                       plot_input = False,
                       check_data=False,
                       axis = 0,
                       direction='left',
                       src = r"f:\TUD_CEG_Repository\MSc_Thesis\Code\run.bat"):
    """
    base = str, name of base model folder
    scenario = str, name of scenario folder
    erosion = bool, modify dem with erosion
    erosion = bool, modify dem with subsidence
    nyear = int, projection year
    geom_file = geojson, file containing model domain polygon
    cl_file = geojson, file containing shoreline with line format or
               'gctr', shoreline from gctr
    use_subgrid = bool, model subgrid
    """

    src_path = Path.cwd().parent / 'base_model'
    model_path = Path.cwd().parent / 'model'

    # base_model = Path(base_model)
    # model_root = scenario
    # new_model = Path(model_root)

    base_model = src_path / base_model
    model_root = model_path / scenario

    

    shutil.copytree(
        base_model,
        model_root,
        dirs_exist_ok=True
    )

    # Initialize logging, the lower the log level number, the more verbose (more info) the output
    # NOTSET=0-9, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
    log.initialize_logging()
    log.set_log_level(log_level=20)

    # Add file handler to log to a file
    log_file = Path(model_root) / "hydromt_sfincs.log"
    logger = log._add_filehandler(log_file)

    # catalog_path = os.path.join(os.getcwd(),'catalog')
    yaml_path = os.path.join(model_path,'data_catalog.yml')
    # yaml_path = os.path.join(model_path,'input.yml')

    sf = SfincsModel(
        data_libs=yaml_path,  # specify which data libraries to use
        root=model_root,  # specify the root directory for the model
        mode="r+",  # specify the mode for opening the model (r=read only, r+=append, w=write, w+=overwrite
        # write_gis=True,  # specify whether to write GIS data
    )

    sf.read()

    res = sf.config.get("dx")
    da = sf.elevation.data

    if plot_input:
        sf.plot_basemap(
            variable="dep", plot_geoms=True, plot_bounds=True, bmap="sat", zoomlevel=12
        )
        sf.plot_forcing()
        

    if erosion or subsidence:

        dep_old = np.copy(da.dep.values)

        # cl_files = r'f:\TUD_CEG_Repository\MSc_Thesis\Code\trial\input\bohai_cl.geojson'
        # dem_crs = sf.crs.to_epsg()
        # geom_files = Path(model_root)/'gis/region.geojson'


        # nyear = 100
        if region is None:
            region = sf.region

        if erosion:
            gctr = get_GCTR_data(region)
            horizontal_shift = shoreline_translation(da,res,gctr,years=nyear,cl_file=cl_file,normal_direction=-1)
            da['dep'].values = horizontal_shift

        if subsidence:
            vertical_shift = subsidence_translation(da,nyear,global_rate=1,geom_files=None)
            da['dep'].values = vertical_shift

        dep_new = np.copy(da.dep.values)
        sf.write()

    if check_data:
        m, n = np.shape(da.dep.values)
        if axis == 0:
            line = m
        else:
            line = n
        # lines = np.linspace(100, m, 4, dtype=int)
        lines = [int(0.1*line),int(0.5*line),int(0.9*line)]
        difference_map(dep_old,dep_new,lines=lines,axis=axis)

        for line in lines:
            profile_check(dep_old,dep_new,res,line,axis=axis,direction='right')


    # if use_subgrid:
    #     catalog_subgrid = create_subgrid_catalog(model_root,da)

    #     sf = SfincsModel(
    #         data_libs=catalog_subgrid,  # specify which data libraries to use
    #         root=model_root,  # specify the root directory for the model
    #         mode="r+",  # specify the mode for opening the model (r=read only, r+=append, w=write, w+=overwrite
    #         write_gis=True,  # specify whether to write GIS data
    #     )

    #     # nr_subgrid = 6
    #     elevation_list = [{"elevation": "large_model_dep"}]

    #     # # Roughness sets
    #     roughness_list = [{"manning": "large_model_manning"}]

    #     sf.subgrid.create(
    #         elevation_list=elevation_list,
    #         roughness_list=roughness_list,
    #         nr_subgrid_pixels=nr_subgrid,
    #         write_dep_tif=True,
    #         write_man_tif=True,
    #     )

    #     sf.write()

    if use_subgrid:
        # catalog_subgrid = create_subgrid_catalog(model_root,da)

        sf = SfincsModel(
            data_libs=yaml_path,  # specify which data libraries to use
            root=model_root,  # specify the root directory for the model
            mode="r+",  # specify the mode for opening the model (r=read only, r+=append, w=write, w+=overwrite
            write_gis=True,  # specify whether to write GIS data
        )

        # nr_subgrid = 6
        elevation_list = [{"elevation": "bohai_10m_eroded"}]

        # # Roughness sets
        # roughness_list = [{"manning": "large_model_manning"}]

        sf.subgrid.create(
            elevation_list=elevation_list,
            # roughness_list=roughness_list,
            manning_land = 0.04,
            manning_sea = 0.02,
            nr_subgrid_pixels=nr_subgrid,
            write_dep_tif=True,
            write_man_tif=True,
        )

        sf.write()

        # NOTE: we turned on that the merged topobathy of the different (high-res) datasets is written to a geotiff

        # uncomment to see the subgrid table variales
        # sf.subgrid.data

        # we can plot the 2D subgrid variables
        # _ = sf.plot_basemap(
        #     variable=sf.subgrid.data["u_navg"], plot_bounds=False, bmap="sat", zoomlevel=12
        # )

    run_simulation(src,model_root)

    da_hmax = post_process_output(model_root,yaml_path)

    results = summarize_flood(sf, da_hmax, scenario,use_subgrid=use_subgrid,nr_subgrid=nr_subgrid)

    return results


def run_manual_scenario(base_model, scenario,
                       erosion = False, subsidence = False, nyear = 50,
                       region = None,
                       cl_file = 'gctr',
                       use_subgrid = True, nr_subgrid = 10,
                       plot_input = False,
                       check_data=False,
                       axis = 0,
                       direction='left',
                       src = r"f:\TUD_CEG_Repository\MSc_Thesis\Code\run.bat"):
    """
    base = str, name of base model folder
    scenario = str, name of scenario folder
    erosion = bool, modify dem with erosion
    erosion = bool, modify dem with subsidence
    nyear = int, projection year
    geom_file = geojson, file containing model domain polygon
    cl_file = geojson, file containing shoreline with line format or
               'gctr', shoreline from gctr
    use_subgrid = bool, model subgrid
    """

    src_path = Path.cwd().parent / 'base_model'
    model_path = Path.cwd().parent / 'model'

    location = base_model.split("_")[0]

    base_model = src_path / base_model
    model_root = model_path / scenario

    shutil.copytree(
        base_model,
        model_root,
        dirs_exist_ok=True
    )

    # Initialize logging, the lower the log level number, the more verbose (more info) the output
    # NOTSET=0-9, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
    log.initialize_logging()
    log.set_log_level(log_level=20)

    # Add file handler to log to a file
    log_file = Path(model_root) / "hydromt_sfincs.log"
    logger = log._add_filehandler(log_file)

    # catalog_path = os.path.join(os.getcwd(),'catalog')
    # yaml_path = os.path.join(model_path,'data_catalog.yml')
    yaml_path = os.path.join(model_path,'input.yml')

    sf = SfincsModel(
        data_libs=yaml_path,  # specify which data libraries to use
        root=model_root,  # specify the root directory for the model
        mode="r+",  # specify the mode for opening the model (r=read only, r+=append, w=write, w+=overwrite
        # write_gis=True,  # specify whether to write GIS data
    )

    sf.read()

    res = sf.config.get("dx")
    da = sf.elevation.data

    if erosion and subsidence:
        elevation_file = f"{location}_erosion_subsidence_{nyear}"
        print(f"Scenario: Erosion + Subsidence ({nyear} yr)")

    elif erosion:
        elevation_file = f"{location}_erosion_{nyear}"
        print(f"Scenario: Erosion ({nyear} yr)")

    elif subsidence:
        elevation_file = f"{location}_subsidence_{nyear}"
        print(f"Scenario: Subsidence ({nyear} yr)")

    else:
        elevation_file = f"{location}_10m"      # instead of hard-coded "bohai_10m"
        print("Scenario: Flood only (10 m resampled DEM)")

    elevation_list = [{"elevation": elevation_file}]
    sf.elevation.create(
        elevation_list=elevation_list,
        buffer_cells=1,
    )

    if plot_input:
        sf.plot_basemap(
            variable="dep", plot_geoms=True, plot_bounds=True, bmap="sat", zoomlevel=12
        )
        sf.plot_forcing()    

    # if check_data:
    #     m, n = np.shape(da.dep.values)
    #     if axis == 0:
    #         line = m
    #     else:
    #         line = n
    #     # lines = np.linspace(100, m, 4, dtype=int)
    #     lines = [int(0.1*line),int(0.5*line),int(0.9*line)]
    #     difference_map(dep_old,dep_new,lines=lines,axis=axis)

    #     for line in lines:
    #         profile_check(dep_old,dep_new,res,line,axis=axis,direction='right')

    if use_subgrid:
        print('Creating subgrid....')
        sf.subgrid.create(
            elevation_list=elevation_list,
            manning_land = 0.04,
            manning_sea = 0.02,
            nr_subgrid_pixels=nr_subgrid,
            write_dep_tif=True,
            write_man_tif=True,
        )



        # NOTE: we turned on that the merged topobathy of the different (high-res) datasets is written to a geotiff

        # uncomment to see the subgrid table variales
        # sf.subgrid.data

        # we can plot the 2D subgrid variables
        # _ = sf.plot_basemap(
        #     variable=sf.subgrid.data["u_navg"], plot_bounds=False, bmap="sat", zoomlevel=12
        # )

    sf.write()

    run_simulation(src,model_root)

    yaml_pp = os.path.join(model_path,'data_catalog.yml')
    da_hmax, _ = post_process_output(model_root,yaml_pp)

    dx = sf.config.get("dx")
    dy = sf.config.get("dy")
    if use_subgrid:
        dx = sf.config.get("dx") / nr_subgrid
        dy = sf.config.get("dy") / nr_subgrid

    results = summarize_flood(da_hmax, scenario, dx, dy)

    return results

# def get_elevation_file(location, scenario, nyear, use_subgrid=True):
#     """
#     Returns the DEM name/path needed by HydroMT.
#     Creates the DEM if it does not exist yet.
#     """

#     base_10m = f"{location}_10m"

#     if scenario == "flood":
#         return base_10m

#     out_name = f"{location}_{scenario}_{nyear}"

#     if not dem_exists(out_name):
#         create_scenario_dem(
#             base_dem=base_10m,
#             scenario=scenario,
#             nyear=nyear,
#             out_name=out_name,
#         )

#     return out_name