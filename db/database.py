import pymongo
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config import _C as cfg_file
from misc.printer import Logger

# Wrapper for MongoDB Database connection
class Database():
    def __init__(self, db_configs, logger:Logger = None):
        # DB Configs
        self.db_configs = db_configs # Config object
        self.connection_string = self.db_configs.URL
        self.verbose = self.db_configs.VERBOSE

        self.logger = logger # Logger object

        self.logger.debug("Connecting to MongoDB API...")
        self.connection  = pymongo.MongoClient(self.connection_string)

        if self.connection is None:
            self.logger.debug("Could not connect to MongoDB API. Check URL")
            raise Exception()
        else:
            self.logger.debug("Connected to MongoDB API")

        self.db = self.connection[self.db_configs.NAME]

        if self.db is None:
            self.logger.debug("Could not connect to ReID DB. Is it present?")
            raise Exception()
        
        self.vehicles_col = self.db[self.db_configs.VEHICLES_COL]
        self.cameras_col = self.db[self.db_configs.CAMERAS_COL]
        self.trajectories_col = self.db[self.db_configs.TRAJECTORIES_COL]
        self.bboxes_col = self.db[self.db_configs.BBOXES_COL]

    def clean(self):
        if self.verbose: self.logger.debug("Cleaning up the database...")

        self.vehicles_col.drop()
        self.cameras_col.drop()
        self.trajectories_col.drop()
        self.bboxes_col.drop()

    def print_status(self):
        if self.verbose:
            self.logger.debug("Database Status:")
            self.logger.debug(f"Vehicles Collection: {self.vehicles_col.count_documents({})} documents")
            self.logger.debug(f"Cameras Collection: {self.cameras_col.count_documents({})} documents")
            self.logger.debug(f"Trajectories Collection: {self.trajectories_col.count_documents({})} documents")
            self.logger.debug(f"BBoxes Collection: {self.bboxes_col.count_documents({})} documents")
            
            # Get DB Size in MB
            db_stats = self.db.command("dbstats")
            self.logger.debug(f"Database Size: {db_stats['dataSize'] / 1024 / 1024:.2f} MB")

    def insert_vehicle(self, vehicle):
        if self.verbose: self.logger.debug(f"Inserting vehicle with ID: {vehicle['_id']}")

        self.vehicles_col.insert_one(vehicle)

    def insert_camera(self, camera):
        if self.verbose:
            self.logger.debug(f"Inserting camera with ID: {camera['_id']}")

        self.cameras_col.insert_one(camera)
    
    def insert_trajectory(self, trajectory, update=False):
        if self.verbose: self.logger.debug(f"Inserting trajectory with ID: {trajectory['_id']}")

        # Update the trajectory if requested
        if update:
            # Extract the necessary fields from the new trajectory data
            new_end_time = trajectory['end_time']

            # Retrieve the new trajectory data
            new_trajectory_data = trajectory['trajectory_data']

            # Prepare the update query
            self.trajectories_col.update_one(
                { '_id': trajectory['_id'], 'vehicle_id': trajectory['vehicle_id']},  # Match the trajectory by its ID
                {
                    # Update the end time
                    "$set": {'end_time': new_end_time},
                    
                    # Append the new trajectory data to the existing array
                    "$push": {'trajectory_data': {"$each": new_trajectory_data}}
                },
            )
        else:
            # Insert the new trajectory if it's not an update
            self.trajectories_col.insert_one(trajectory)

    def insert_bbox(self, bbox):
        if self.verbose: self.logger.debug(f"Inserting bounding box with ID: {bbox['_id']}")

        self.bboxes_col.insert_one(bbox) # Insert the new bounding box

    def update_bbox(self, new_vid, old_vid):
        # Find all documents with the old vehicle_id and new vehicle_id
        old_documents = list(self.bboxes_col.find({'vehicle_id': old_vid}))
        new_documents = list(self.bboxes_col.find({'vehicle_id': new_vid}))

        # Combine both sets of documents and sort them by timestamp
        combined_documents = sorted(old_documents + new_documents, key=lambda x: (x['timestamp'], x['frame_number']))

        # Prepare a list of update operations
        updates = []
        deletes = []

        for i, doc in enumerate(combined_documents):
            # Create the new _id ensuring frame numbers are sequential
            new_id = f"{new_vid}_F{i+1}"

            # Update the document to have the correct new_vid and _id
            update_operation = pymongo.InsertOne(
                {
                    '_id': new_id,                      # New sequential ID
                    'frame_number': doc['frame_number'], # Updated frame number
                    'vehicle_id': new_vid,              # Updated vehicle ID
                    'compressed_image': doc['compressed_image'],
                    'bounding_box': doc['bounding_box'],
                    'confidence': doc['confidence'],
                    'features': doc['features'],
                    'shape': doc['shape'],
                    'timestamp': doc['timestamp'],
                }
            )
            updates.append(update_operation)
            deletes.append(pymongo.DeleteOne({'_id': doc['_id']}))  # Delete the old document

        # Execute the updates in bulk
        if deletes:
            self.bboxes_col.bulk_write(deletes)
            if self.verbose:
                self.logger.debug(
                    f"Deleted {len(deletes)} documents. Merged vehicle {old_vid} into {new_vid}.")
        if updates:
            self.bboxes_col.bulk_write(updates)
            if self.verbose:
                self.logger.debug(
                    f"Updated {len(combined_documents)} documents. Merged vehicle {old_vid} into {new_vid}."
                )
        else:
            if self.verbose:
                self.logger.debug("No documents found to update.")

    def update_trajectory(self, new_vid, old_vid):
        # Step 1: Retrieve the trajectory for the new vehicle
        trajectory = self.trajectories_col.find_one({'vehicle_id': new_vid})
        trajectory_old = self.trajectories_col.find_one({'vehicle_id': old_vid})
        
        if not trajectory or not trajectory_old:
            self.logger.debug(f"No trajectory found for vehicle {new_vid}")
            return

        # Update trajectory with the new start and end time (start time must be the minimum of the two)
        new_start_time = min(trajectory['start_time'], trajectory_old['start_time'])
        new_end_time = max(trajectory['end_time'], trajectory_old['end_time'])

        # Step 8: Apply the update to the trajectory
        self.trajectories_col.update_one({'_id': trajectory['_id']},
                                            {"$set": {
                                            'start_time': new_start_time,
                                            'end_time': new_end_time}},
                                            upsert=True)
        
        # Step 9: Delete the old vehicle trajectory
        self.trajectories_col.delete_one({'vehicle_id': old_vid})
                
        if self.verbose: self.logger.debug(f"Updated trajectory for Vehicle {new_vid} with info coming from Vehicle {old_vid}.")

    def remove_vehicle(self, vehicle_id):
        if self.verbose: self.logger.debug(f"Removing vehicle with ID: {vehicle_id}")

        self.vehicles_col.delete_one({'_id': vehicle_id})

    def get_vehicle_frames(self, vehicle_id):
        db_frames = self.trajectories_col.aggregate([
        {'$match': {'vehicle_id': vehicle_id}},
        {
            '$project': {
                '_id': 0, 
                'vehicle_id': 1, 
                'camera_id': 1,
                # "trajectory_data._id": 1,
                # 'trajectory_data.compressed_image': 1,
                # 'trajectory_data.shape': 1, 
                # 'trajectory_data.features': 1
            }
        }])

        # Dictionary to store the results
        frames_dict = {}

        # Iterate over the query results and construct the dictionary
        for x in db_frames:
            vehicle_id = x['vehicle_id']
            #camera_id = x['camera_id']

            # from the trajectory_data._id, we must query the bbox collection to get the compressed_image, shape, and features
            trajectory_data = self.bboxes_col.aggregate([
                {'$match': {'vehicle_id': vehicle_id}},
                {
                    '$project': {
                        '_id': 1,
                        'compressed_image': 1,
                        'shape': 1,
                        'features': 1
                    }
                }
            ])

            # Create a dictionary for each vehicle, mapping bbox_id to features
            frames_dict[vehicle_id] = {data['_id']: [data['features'], data['compressed_image'], data['shape']] for data in trajectory_data}

        return frames_dict

    def get_camera_frames(self, camera_id):

        # Dictionary to store the results
        frames_dict = {}

        db_frames = self.trajectories_col.aggregate([
        {'$match': {'camera_id': camera_id}},
        {
            '$project': {
                '_id': 1, 
                'vehicle_id': 1, 
                'camera_id': 1,
                # 'trajectory_data.compressed_image': 1,
                # 'trajectory_data.shape': 1, 
                # 'trajectory_data.features': 1
            }
        }])

        # Iterate over the query results and construct the dictionary
        for i, x in enumerate(db_frames):
            vehicle_id = x['vehicle_id']
            #camera_id = x['camera_id']

            # from the trajectory_data._id, we must query the bbox collection to get the compressed_image, shape, and features
            traj_data = self.bboxes_col.aggregate([
                {'$match': {'vehicle_id': vehicle_id}},
                {
                    '$project': {
                        '_id': 1,
                        'compressed_image': 1,
                        'shape': 1,
                        'features': 1,
                        'frame_number': 1,
                        'bounding_box': 1,
                        'timestamp': 1,
                        'confidence': 1,
                        'vehicle_id': 1
                    }
                }
            ])

            # Create a dictionary for each vehicle, mapping bbox_id to features
            frames_dict[vehicle_id] = {data['_id']: [data['features'], data['compressed_image'], data['shape'],
                                                     data['frame_number'], data['bounding_box'], data['timestamp'],
                                                     data['confidence'], data['vehicle_id']] for data in traj_data}

        return frames_dict
    
    def get_all_trajectories(self, camera_id):
        db_frames = self.trajectories_col.aggregate([
            {'$match': {'camera_id': camera_id}},
            {
                '$project': {
                    '_id': 1, 
                    'vehicle_id': 1, 
                    'camera_id': 1,
                    'start_time': 1,
                    'end_time': 1,
                    # "trajectory_data._id": 1,
                    # 'trajectory_data.compressed_image': 1,
                    # 'trajectory_data.shape': 1, 
                    # 'trajectory_data.features': 1,
                    # 'trajectory_data.bounding_box': 1
                }
            }
        ])

        # Dictionary to store the results
        frames_dict = {}

        # Iterate over the query results and construct the dictionary
        for x in db_frames:
            vehicle_id = x['vehicle_id']

            # from the trajectory_data._id, we must query the bbox collection to get the compressed_image, shape, and features
            traj_data = self.bboxes_col.aggregate([
                {'$match': {'vehicle_id': vehicle_id}},
                {
                    '$project': {
                        '_id': 1,
                        'compressed_image': 1,
                        'shape': 1,
                        'features': 1,
                        'bounding_box': 1
                    }
                }
            ])

            # Create a dictionary for each vehicle, mapping bbox_id to features
            if vehicle_id not in frames_dict:
                frames_dict[vehicle_id] = {}

            # Modify this part in your code
            frames_dict[vehicle_id] = {data['_id']: {
                'features': data['features'],
                'compressed_image': data['compressed_image'],
                'shape': data['shape'],
                'bounding_box': data['bounding_box']
            } for data in traj_data}

        return frames_dict

    def get_vehicle_single_trajectory(self, vehicle_id):
        db_frames = self.trajectories_col.aggregate([
            {'$match': {'vehicle_id': vehicle_id}},
            {
                '$project': {
                    '_id': 1, 
                    'vehicle_id': 1, 
                    'camera_id': 1,
                    'start_time': 1,
                    'end_time': 1,
                    # "trajectory_data._id": 1,
                    # 'trajectory_data.features': 1,
                    # 'trajectory_data.compressed_image': 1,
                    # 'trajectory_data.shape': 1, 
                    # 'trajectory_data.bounding_box': 1,
                }
            }
        ])

        # THIS ONLY RETURNS ONE SINGLE TRAJECTORY!
        for x in db_frames:
            vehicle_id = x['vehicle_id']
            camera_id = x['camera_id']
            start_time = x['start_time']
            end_time = x['end_time']
            # trajectory_data = x['trajectory_data']            

            # from the trajectory_data._id, we must query the bbox collection to get the compressed_image, shape, and features
            trajectory_data = self.bboxes_col.aggregate([
                {'$match': {'vehicle_id': vehicle_id}},
                {
                    '$project': {
                        '_id': 1,
                        'compressed_image': 1,
                        'shape': 1,
                        'features': 1,
                        'bounding_box': 1,
                        'timestamp': 1
                    }
                }
            ])

            return {'vehicle_id': vehicle_id, 'camera_id': camera_id,
                    'start_time': start_time, 'end_time': end_time,
                    'trajectory_data': {data['_id']: [data['features'], data['compressed_image'], data['shape'], data['bounding_box'], data['timestamp']] for data in trajectory_data}}

    def get_vehicle_trajectory(self, vehicle_id):
        db_frames = self.trajectories_col.aggregate([
            {'$match': {'vehicle_id': vehicle_id}},
            {
                '$project': {
                    '_id': 1, 
                    'vehicle_id': 1, 
                    'camera_id': 1,
                    'start_time': 1,
                    'end_time': 1,
                    # "trajectory_data._id": 1,
                    # 'trajectory_data.features': 1,
                    # 'trajectory_data.compressed_image': 1,
                    # 'trajectory_data.shape': 1, 
                    # 'trajectory_data.bounding_box': 1,
                }
            }
        ])

        # Dictionary to store the results
        frames_dict = {}

        # Iterate over the query results and construct the dictionary
        if len(db_frames._data) == 0 or db_frames._data is None:
            raise Exception(f"No trajectory found for vehicle {vehicle_id}")
        
        # THIS RETURNS MULTIPLE TRAJECTORIES!
        for x in db_frames:
            traj_id = x['_id']
            vehicle_id = x['vehicle_id']
            camera_id = x['camera_id']
            start_time = x['start_time']
            end_time = x['end_time']

            # from the trajectory_data._id, we must query the bbox collection to get the compressed_image, shape, and features
            trajectory_data = self.bboxes_col.aggregate([
                {'$match': {'vehicle_id': vehicle_id}},
                {
                    '$project': {
                        '_id': 1,
                        'compressed_image': 1,
                        'shape': 1,
                        'features': 1,
                        'bounding_box': 1
                    }
                }
            ])

            if vehicle_id not in frames_dict:
                frames_dict[traj_id] = {}

            # Create a dictionary for each camera and vehicle
            frames_dict[traj_id] = {'vehicle_id': vehicle_id,
                                    'camera_id': camera_id,
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'trajectory_data': {data['_id']: [data['features'], data['compressed_image'], data['shape'], data['bounding_box']] for data in trajectory_data}}

            return frames_dict

    def get_all_frames(self):

        # Dictionary to store the results
        frames_dict = {}

        db_frames = self.trajectories_col.aggregate([
            {
                '$project': {
                    '_id': 0, 
                    'vehicle_id': 1, 
                    'camera_id': 1,
                }
            }
        ])

        # Iterate over the query results and construct the dictionary
        for x in db_frames:
            vehicle_id = x['vehicle_id']
            camera_id = x['camera_id']

            # from the trajectory_data._id, we must query the bbox collection to get the compressed_image, shape, and features
            trajectory_data = self.bboxes_col.aggregate([
                {'$match': {'vehicle_id': vehicle_id}},
                {
                    '$project': {
                        '_id': 1,
                        'frame_number': 1,
                        'compressed_image': 1,
                        'shape': 1,
                        'features': 1,
                        'timestamp': 1,
                        'bounding_box': 1
                    }
                }
            ])

            # Create a dictionary for each camera and vehicle, mapping bbox_id to features
            if camera_id not in frames_dict:
                frames_dict[camera_id] = {}

            # Initialize the vehicle entry in the camera if it doesn't exist
            if vehicle_id not in frames_dict[camera_id]:
                frames_dict[camera_id][vehicle_id] = {}

            for data in trajectory_data:
                # Append the frame data to the vehicle entry for this camera
                frames_dict[camera_id][vehicle_id].update(
                    {data['frame_number']: {
                        'frame_number': data['frame_number'],
                        'bounding_box': data['bounding_box'],
                        'features': data['features'],
                        'compressed_image': None,#data['compressed_image'],
                        'shape': data['shape'],
                        'timestamp': data['timestamp'],
                        }
                    }
                )

        return frames_dict
    
    def get_camera(self, camera_id):
        camera = self.cameras_col.find_one({'_id': camera_id})
        return camera

if __name__ == '__main__':
    # Get an instance of the logger
    logger = Logger()
    
    # Create a new database object
    cfg_file.DB.VERBOSE = True
    db = Database(db_configs=cfg_file.DB, logger=logger)

    # Print the database status
    db.print_status()
