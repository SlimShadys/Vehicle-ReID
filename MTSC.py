import argparse
import os
import sys

import yaml
from misc.printer import Logger
from pipeline import load_config_and_device, load_models_and_transformations, \
                    run_mtsc, setup_database

def main(camera_data):
    # Initialize logger
    logger = Logger()

    # Load configuration and initialize device
    cfg, device = load_config_and_device()
    
    # Load models and transformations
    yolo_model, model, car_classifier, transforms = load_models_and_transformations(device, cfg)
    
    # Setup database (Cleaning is inside the function)
    if cfg.DB.USE_DB:
        db = setup_database()
    else:
        db = None

    # Populate camera data into cfg_file.CAMERA
    _, camera_data = next(iter(camera_data.items()))  # Access the single key-value pair in camera_data

    info_path = camera_data.get('info', None)
    if info_path and os.path.exists(info_path):
        with open(info_path, 'r') as file:
            layout = yaml.safe_load(file)
        # Initialize missing fields to avoid KeyError
        camera_data['_id'] = layout.get('_id', None)
        camera_data['name'] = layout.get('name', None)
        camera_data['location'] = layout.get('location', None)
        camera_data['coordinates'] = layout.get('coordinates', None)
        camera_data['description'] = layout.get('description', None)
    else:
        raise FileNotFoundError(f"File {info_path} does not exist. You should provide a minimum of information for the camera.")

    if cfg.DB.USE_DB:
        # We need a way to give the camera ID the most updated index when _id is None.
        if camera_data['_id'] is None:
            # Get the last camera ID and increment it by 1
            last_camera_id = list(db.cameras_col.aggregate([
                {'$sort': {'_id': -1}},
                {'$limit': 1},
                {'$project': {'_id': 1}}
            ]))

            if last_camera_id:
                camera_data['_id'] = last_camera_id[0]['_id'] + 1
            else:
                # Default to 1 if no cameras are present in the collection
                camera_data['_id'] = 1
    else:
        camera_data['_id'] = 1

    if camera_data['name'] is None:
        camera_data['name'] = f"Camera-{camera_data['_id']}"

    if cfg.DB.USE_DB:
        # Here we need to make sure that the camera is not already in the database
        if db.cameras_col.find_one({'_id': camera_data['_id']}) is not None:
            logger.error(f"Camera with ID {camera_data['_id']} is already in the database.")
            #sys.exit(1)
        else:
            logger.info("Inserting Camera into the database...")
            db.insert_camera(camera_data)
        
    logger.info(f"Successfully set camera with ID: {camera_data['_id']}")

    # Run the pipeline for the single camera
    run_mtsc(yolo_model, model, car_classifier, transforms, device, camera_data)

if __name__ == '__main__':

    print("|***************************************************|")
    print("|                  Vehicle Re-ID                    |")
    print("|                   Single MTSC                     |")
    print("|---------------------------------------------------|")
    print("|             Made by: Gianmarco Scarano            |")
    print("|       --- MSc Student at AI & Robotics ---        |")
    print("|      --- University of Rome, La Sapienza ---      |")
    print("|         ---        Rome, Italy        ---         |")
    print("|***************************************************|")

    parser = argparse.ArgumentParser(description="Run MTSC for a single camera.")
    parser.add_argument("--config", required=False, help="Path to the camera configuration file (YAML format).")
    args = parser.parse_args()

    camera_config = args.config if args.config else 'configs\\camera_s02_cityflow.yml'

    # Load camera data from YAML file
    with open(camera_config, 'r') as f:
        camera_data = yaml.safe_load(f)['MTSC']
    
    main(camera_data)
