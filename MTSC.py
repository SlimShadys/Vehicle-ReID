import argparse
import os

import yaml

from evaluate_AICity import run_evaluation
from misc.printer import Logger
from pipeline import (load_config_and_device, load_models_and_transformations,
                      run_mtsc, setup_database)

def main(camera):
    # Load configuration and initialize device
    cfg, device = load_config_and_device()

    # Get Test name
    test_name = cfg.MISC.TEST_NAME
    
    # Initialize logger
    logger = Logger(name=test_name)
  
    # Load models and transformations
    yolo_model, model, car_classifier, transforms = load_models_and_transformations(device, cfg)
    
    # Setup database (Cleaning is inside the function)
    db = setup_database()

    # Populate camera data into cfg_file.CAMERA
    camera_configs = camera['MTSC']
    camera_name, camera_data = next(iter(camera_configs.items()))  # Access the single key-value pair in camera_data

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

    # We need a way to give the camera ID the most updated index when _id is None.
    if camera_data['_id'] is None:
        if db is not None:
            # Get the last camera ID and increment it by 1
            last_camera_id = db.cameras_col.aggregate([
                {'$sort': {'_id': -1}},
                {'$limit': 1},
                {'$project': {'_id': 1}}
            ])._data[0]['_id']

            camera_data['_id'] = last_camera_id + 1
        else:
            logger.error("Camera ID is None and no database is available to get the last camera ID.")
            raise ValueError()
        
    if camera_data['name'] is None:
        camera_data['name'] = f"Camera-{camera_data['_id']}"

    # Here we need to make sure that the camera is not already in the database
    if db is not None:
        if db.cameras_col.find_one({'_id': camera_data['_id']}) is not None:
            logger.error(f"Camera with ID {camera_data['_id']} is already in the database.")
            #raise ValueError()
        else:
            logger.debug("Inserting Camera into the database...")
            db.insert_camera(camera_data)

    logger.info(f"Successfully found and set camera with ID: {camera_data['_id']}")

    # Run the pipeline for the single camera
    _ = run_mtsc(yolo_model, model, car_classifier, transforms, device, camera_data, db, logger)

    if cfg.MISC.EVALUATE_METRICS:
        logger.info("Evaluating MTSC results...")

        # Retrieve the MTSC file files and place it in cfg.METRICS.PREDICTIONS
        cam_num = int(camera_name.split("_")[-1])
        cfg.METRICS.PREDICTIONS = [f for f in os.listdir() if f.startswith(f'{test_name}-MTSC-predictions_camera-{cam_num}') and f.endswith('.txt')]

        results = run_evaluation(cfg, camera_configs)
        logger.info("MTSC Evaluation results:\n" + results + "\n")

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

    camera_config = args.config if args.config else os.path.join("configs", "camera_s02_cityflow.yml")

    # Load camera data from YAML file
    with open(camera_config, 'r') as f:
        camera = yaml.safe_load(f)
    
    main(camera)
