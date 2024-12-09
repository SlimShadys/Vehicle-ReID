import argparse
import os

import numpy as np
import yaml
from evaluate_AICity import run_evaluation
from misc.printer import Logger
from pipeline import load_config_and_device, load_models_and_transformations, \
                    run_mtsc, run_mtmc, setup_database

def main(cam_configs):
    # Initialize logger
    logger = Logger()
    
    # Load configuration and initialize device
    cfg, device = load_config_and_device()
    
    # Load models and transformations
    yolo_model, model, car_classifier, transforms = load_models_and_transformations(device, cfg)

    # Setup database
    db = setup_database()

    # Setup Camera configs and Layout
    camera_configs = cam_configs['MTMC']
    camera_layout = cam_configs['layout']

    # Init performance dict
    performance = {}

    # Run single-camera pipeline for each camera in config
    for camera_name, camera_data in camera_configs.items():
        info_path = camera_data.get('info', None)
        if info_path and os.path.exists(info_path):
            with open(info_path, 'r') as file:
                layout = yaml.safe_load(file)
                
            camera_data['_id'] = layout.get('_id', None)
            camera_data['name'] = layout.get('name', None)
            camera_data['location'] = layout.get('location', None)
            camera_data['coordinates'] = layout.get('coordinates', None)
            camera_data['description'] = layout.get('description', None)

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
        logger.info(f"Processing {camera_name}...")
        
        # Run the pipeline for the single camera
        records = run_mtsc(yolo_model, model, car_classifier, transforms, device, camera_data, db, logger)
        
        # Delete first entry of records (Header)
        records = records[1:]

        for record in records:
            name = record[0]
            min_t = round(float(record[1]), 4)
            mean_t = round(float(record[2]), 4)
            max_t = round(float(record[3]), 4)

            if name not in performance:
                performance[name] = []
            
            performance[name].append([min_t, mean_t, max_t])

        # # Reset the YOLO Tracker stored IDs for the next camera
        # yolo_model.predictor.trackers[0].reset()

        logger.info("************************************************")

    # Run cross-camera analysis after all single-camera pipelines
    run_mtmc(
        similarity_method=cfg.REID.TEST.SIMILARITY_METHOD,
        db=db,
        cam_layout=camera_layout,
        cfg=cfg,
        logger=logger
    )

    logger.info("MTMC Pipeline completed successfully.")

    # Print the overall mean of the performance results over all the videos
    if len(performance) > 0:
        logger.info("Performance results:")
        max_len = max([len(name) for name, _ in performance.items()])

        for name, values in performance.items():
            # Compute statistics
            min_val = round(np.mean([x[0] for x in values]), 4)
            mean_val = round(np.mean([x[1] for x in values]), 4)
            max_val = round(np.mean([x[2] for x in values]), 4)

            # Adjust the name width dynamically using max_len
            formatted_name = name.ljust(max_len)
            
            # Log the formatted string
            logger.info(f"Performance for {formatted_name} | MIN: {min_val:.4f} | MEAN: {mean_val:.4f} | MAX: {max_val:.4f}")

    if cfg.MISC.EVALUATE_METRICS:
        logger.info("Evaluating MTMC results...")

        # Retrieve the MTMC files and place them in cfg.METRICS.PREDICTIONS
        mtmc_files = [f for f in os.listdir() if f.startswith('MTMC-predictions_camera') and f.endswith('.txt')]
        cfg.METRICS.PREDICTIONS = mtmc_files

        results = run_evaluation(cfg, camera_configs)
        logger.info("MTMC Evaluation results:\n" + results + "\n")

if __name__ == '__main__':

    print("|***************************************************|")
    print("|                   Vehicle Re-ID                   |")
    print("|                      Pipeline                     |")
    print("|---------------------------------------------------|")
    print("|             Made by: Gianmarco Scarano            |")
    print("|       --- MSc Student at AI & Robotics ---        |")
    print("|      --- University of Rome, La Sapienza ---      |")
    print("|         ---        Rome, Italy        ---         |")
    print("|***************************************************|")

    parser = argparse.ArgumentParser(description="Run MTMC for multiple cameras.")
    parser.add_argument("--config", required=False, help="Path to the cameras configuration file (YAML format).")
    args = parser.parse_args()

    camera_configs = args.config if args.config else 'configs\\cameras_s02_cityflow.yml'

    # Load cameras data from YAML file
    with open(camera_configs, 'r') as f:
        camera_configs = yaml.safe_load(f)
    
    main(camera_configs)
