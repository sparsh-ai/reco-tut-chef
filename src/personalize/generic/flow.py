interactions_file_path = './data/silver/ml-latest-small/interactions.csv'
interactions_filename = 'interactions.csv'
boto3.Session().resource('s3').Bucket(bucket_name).Object(interactions_filename).upload_file(interactions_file_path)
interactions_s3DataPath = "s3://"+bucket_name+"/"+interactions_filename