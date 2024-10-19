import pymongo
from misc.printer import Logger

# Wrapper for MongoDB Database connection
class Database():
    def __init__(self, db_configs, logger:Logger = None):
        # DB Configs
        self.db_configs = db_configs # Config object
        self.connection_string = self.db_configs.URL
        self.verbose = self.db_configs.VERBOSE

        self.logger = logger # Logger object

        self.logger.db("Connecting to MongoDB API...")
        self.connection  = pymongo.MongoClient(self.connection_string)

        if self.connection is None:
            self.logger.db("Could not connect to MongoDB API. Check URL")
            raise Exception()
        else:
            self.logger.db("Connected to MongoDB API")

        self.db = self.connection[self.db_configs.NAME]

        if self.db is None:
            self.logger.db("Could not connect to ReID DB. Is it present?")
            raise Exception()
        
        self.vehicles_col = self.db[self.db_configs.VEHICLES_COL]
        self.cameras_col = self.db[self.db_configs.CAMERAS_COL]
        self.trajectories_col = self.db[self.db_configs.TRAJECTORIES_COL]
        self.bboxes_col = self.db[self.db_configs.BBOXES_COL]

    def clean(self):
        if self.verbose: self.logger.db("Cleaning up the database...")

        self.vehicles_col.drop()
        self.cameras_col.drop()
        self.trajectories_col.drop()
        self.bboxes_col.drop()

    def insert_vehicle(self, vehicle):
        if self.verbose: self.logger.db(f"Inserting vehicle with ID: {vehicle['_id']}")

        self.vehicles_col.insert_one(vehicle)

    def insert_camera(self, camera):
        if self.verbose:
            self.logger.db(f"Inserting camera with ID: {camera['_id']}")

        self.cameras_col.insert_one(camera)
    
    def insert_trajectory(self, trajectory, update=False):
        if self.verbose: self.logger.db(f"Inserting trajectory with ID: {trajectory['_id']}")

        # Extract the necessary fields from the new trajectory data
        new_end_time = trajectory['end_time']
        new_trajectory_data = trajectory['trajectory_data']

        # Update the trajectory if requested
        if update:
            # Prepare the update query
            self.trajectories_col.update_one(
                {'trajectory_id': trajectory['_id']},  # Match the trajectory by its ID
                {
                    # Update the end time
                    "$set": {'end_time': new_end_time},
                    
                    # Append the new trajectory data to the existing array
                    "$push": {'trajectory_data': {"$each": new_trajectory_data}}
                },
                upsert=True  # Insert if the trajectory does not exist
            )
        else:
            # Insert the new trajectory if it's not an update
            self.trajectories_col.insert_one(trajectory)

    def insert_bbox(self, bbox):
        if self.verbose: self.logger.db(f"Inserting bounding box with ID: {bbox['_id']}")

        self.bboxes_col.insert_one(bbox) # Insert the new bounding box

    def update_bbox(self, new_vid, old_vid):
        # Find all documents with the old vehicle_id
        documents = self.bboxes_col.find({'vehicle_id': old_vid})

        # Get latest frame_number from the new vehicle_id and increment it by 1
        frame_number = int([data for data in self.bboxes_col.find({'vehicle_id': new_vid})][-1]['_id'].split('_F')[-1]) + 1

        # Prepare a list of update operations
        updates = []

        for i, doc in enumerate(documents):
            # Create the new _id by replacing the old vehicle ID with the new one
            new_id = f"{new_vid}_F{frame_number + i}"

            # Create an update operation to change both _id and vehicle_id
            update_operation = pymongo.InsertOne(
                {'_id'          : new_id,       # Update the _id to the new one
                'confidence'    : doc['confidence'],
                'bounding_box'  : doc['bounding_box'],
                'cropped_image' : doc['cropped_image'],
                'timestamp'     : doc['timestamp'],
                'features'      : doc['features'],
                'shape'         : doc['shape'],
                'vehicle_id'    : new_vid,      # Update the vehicle_id to the new one
                 }
            )
            updates.append(update_operation)                        # Append the update operation to the list
            updates.append(pymongo.DeleteOne({'_id': doc['_id']}))  # Delete the old document
                                                  
        # Execute the updates in bulk
        if updates:
            self.bboxes_col.bulk_write(updates)
            if self.verbose: self.logger.db(f"Updated {len(updates)} documents. Merged vehicle {old_vid} into {new_vid}.")
        else:
            if self.verbose: self.logger.db("No documents found to update.")

    def update_trajectory(self, new_vid, old_vid):
        # Step 1: Retrieve the trajectory for the new vehicle
        trajectory = self.trajectories_col.find_one({'vehicle_id': new_vid})
        
        if not trajectory:
            self.logger.db(f"No trajectory found for vehicle {new_vid}")
            return

        # Step 2: Get all bounding boxes for the new vehicle from the bboxes collection
        bboxes = list(self.bboxes_col.find({'vehicle_id': new_vid}).sort('timestamp', pymongo.ASCENDING))

        # Step 3: Compare the number of entries in trajectory_data and the bounding box collection
        trajectory_data = trajectory['trajectory_data']
        
        if len(bboxes) == len(trajectory_data):
            self.logger.db(f"Trajectory for {new_vid} is already up-to-date.")
            return
        else:
            # Step 4: Identify missing bounding boxes by timestamp and append them
            bbox_ids_in_trajectory = {box['_id'] for box in trajectory_data}
            new_bboxes = [bbox for bbox in bboxes if bbox['_id'] not in bbox_ids_in_trajectory]

            # Add the missing bounding boxes to trajectory_data
            for bbox in new_bboxes:
                trajectory_data.append({
                    'confidence': bbox['confidence'],
                    'bounding_box': bbox['bounding_box'],
                    'cropped_image': bbox['cropped_image'],
                    'timestamp': bbox['timestamp'],
                    'features': bbox['features'],
                    'shape': bbox['shape'],
                    '_id': bbox['_id'],
                    'vehicle_id': bbox['vehicle_id']
                })

            # Step 5: Sort trajectory data by timestamp
            trajectory_data.sort(key=lambda x: x['timestamp'])

            # Step 6: Update the start and end times
            new_start_time = trajectory_data[0]['timestamp']
            new_end_time = trajectory_data[-1]['timestamp']

            # Step 7: Apply the update to the trajectory
            self.trajectories_col.update_one({'_id': trajectory['_id']},
                                             {"$set": {
                                                'trajectory_data': trajectory_data,
                                                'start_time': new_start_time,
                                                'end_time': new_end_time}},
                                                upsert=True)
            
            # Step 8: Delete the old vehicle trajectory
            self.trajectories_col.delete_one({'vehicle_id': old_vid})
                        
            if self.verbose: self.logger.db(f"Updated trajectory for vehicle {new_vid} with {len(new_bboxes)} new bounding boxes from vehicle {old_vid}.")

    def remove_vehicle(self, vehicle_id):
        if self.verbose: self.logger.db(f"Removing vehicle with ID: {vehicle_id}")

        self.vehicles_col.delete_one({'_id': vehicle_id})

    def get_vehicle_frames(self, vehicle_id):
        db_frames = self.trajectories_col.aggregate([
        {'$match': {'vehicle_id': vehicle_id}},
        {
            '$project': {
                '_id': 0, 
                'vehicle_id': 1, 
                'camera_id': 1,
                "trajectory_data._id": 1,
                'trajectory_data.cropped_image': 1,
                'trajectory_data.shape': 1, 
                'trajectory_data.features': 1
            }
        }])

        # Dictionary to store the results
        frames_dict = {}

        # Iterate over the query results and construct the dictionary
        for x in db_frames:
            vehicle_id = x['vehicle_id']
            trajectory_data = x['trajectory_data']
            #camera_id = x['camera_id']

            # Create a dictionary for each vehicle, mapping bbox_id to features
            frames_dict[vehicle_id] = {data['_id']: [data['features'], data['cropped_image'], data['shape']] for data in trajectory_data}

        return frames_dict

    def get_camera_frames(self, camera_id):
        db_frames = self.trajectories_col.aggregate([
        {'$match': {'camera_id': camera_id}},
        {
            '$project': {
                '_id': 0, 
                'vehicle_id': 1, 
                'camera_id': 1,
                "trajectory_data._id": 1,
                'trajectory_data.cropped_image': 1,
                'trajectory_data.shape': 1, 
                'trajectory_data.features': 1
            }
        }])

        # Dictionary to store the results
        frames_dict = {}

        # Iterate over the query results and construct the dictionary
        for x in db_frames:
            vehicle_id = x['vehicle_id']
            trajectory_data = x['trajectory_data']
            #camera_id = x['camera_id']

            # Create a dictionary for each vehicle, mapping bbox_id to features
            frames_dict[vehicle_id] = {data['_id']: [data['features'], data['cropped_image'], data['shape']] for data in trajectory_data}

        return frames_dict
    
    def get_all_frames(self):
        db_frames = self.trajectories_col.aggregate([
            {
                '$project': {
                    '_id': 0, 
                    'vehicle_id': 1, 
                    'camera_id': 1,
                    "trajectory_data._id": 1,
                    'trajectory_data.cropped_image': 1,
                    'trajectory_data.shape': 1, 
                    'trajectory_data.features': 1
                }
            }
        ])

        # Dictionary to store the results
        frames_dict = {}

        # Iterate over the query results and construct the dictionary
        for x in db_frames:
            vehicle_id = x['vehicle_id']
            camera_id = x['camera_id']
            trajectory_data = x['trajectory_data']

            # Create a dictionary for each camera and vehicle, mapping bbox_id to features
            if camera_id not in frames_dict:
                frames_dict[camera_id] = {}

            frames_dict[camera_id][vehicle_id] = {data['_id']: [data['features'], data['cropped_image'], data['shape']] for data in trajectory_data}

        return frames_dict