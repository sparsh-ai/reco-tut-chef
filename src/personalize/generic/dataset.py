import boto3
import json
import time


class personalize_dataset:
    def __init__(self,
                 dataset_group_arn=None,
                 schema_arn=None,
                 dataset_arn=None,
                 dataset_type='INTERACTIONS',
                 region='us-east-1',
                 bucket_name=None,
                 role_arn=None,
                 source_data_path=None,
                 target_file_name=None,
                 dataset_import_job_arn=None
                 ):
        self.personalize = None
        self.personalize_runtime = None
        self.s3 = None
        self.iam = None
        self.dataset_group_arn = dataset_group_arn
        self.schema_arn = schema_arn
        self.dataset_arn = dataset_arn
        self.dataset_type = dataset_type
        self.region = region
        self.bucket_name = bucket_name
        self.role_arn = role_arn
        self.source_data_path = source_data_path
        self.target_file_name = target_file_name
        self.dataset_import_job_arn = dataset_import_job_arn

    def setup_connection(self):
        try:
            self.personalize = boto3.client('personalize')
            self.personalize_runtime = boto3.client('personalize-runtime')
            self.s3 = boto3.client('s3')
            self.iam = boto3.client("iam")
            print("SUCCESS | We can communicate with Personalize!")
        except:
            print("ERROR | Connection can't be established!")
    
    def create_dataset_group(self, dataset_group_name=None):
        """
        The highest level of isolation and abstraction with Amazon Personalize
        is a dataset group. Information stored within one of these dataset groups
        has no impact on any other dataset group or models created from one. they
        are completely isolated. This allows you to run many experiments and is
        part of how we keep your models private and fully trained only on your data.
        """
        create_dataset_group_response = self.personalize.create_dataset_group(name=dataset_group_name)
        self.dataset_group_arn = create_dataset_group_response['datasetGroupArn']
        # print(json.dumps(create_dataset_group_response, indent=2))

        # Before we can use the dataset group, it must be active. 
        # This can take a minute or two. Execute the cell below and wait for it
        # to show the ACTIVE status. It checks the status of the dataset group
        # every minute, up to a maximum of 3 hours.
        max_time = time.time() + 3*60*60 # 3 hours
        while time.time() < max_time:
            status = self.check_dataset_group_status()
            print("DatasetGroup: {}".format(status))
            if status == "ACTIVE" or status == "CREATE FAILED":
                break
            time.sleep(60)

    def check_dataset_group_status(self):
        """
        Check the status of dataset group
        """
        describe_dataset_group_response = self.personalize.describe_dataset_group(
            datasetGroupArn = self.dataset_group_arn
            )
        status = describe_dataset_group_response["datasetGroup"]["status"]
        return status

    def create_dataset(self, schema=None, schema_name=None, dataset_name=None):
        """
        First, define a schema to tell Amazon Personalize what type of dataset
        you are uploading. There are several reserved and mandatory keywords
        required in the schema, based on the type of dataset. More detailed
        information can be found in the documentation.
        """
        create_schema_response = self.personalize.create_schema(
            name = schema_name,
            schema = json.dumps(schema)
        )
        self.schema_arn = create_schema_response['schemaArn']

        """
        With a schema created, you can create a dataset within the dataset group.
        Note that this does not load the data yet, it just defines the schema for
        the data. The data will be loaded a few steps later.
        """
        create_dataset_response = self.personalize.create_dataset(
            name = dataset_name,
            datasetType = self.dataset_type,
            datasetGroupArn = self.dataset_group_arn,
            schemaArn = self.schema_arn
        )
        self.dataset_arn = create_dataset_response['datasetArn']
    
    def create_s3_bucket(self):
        if region == "us-east-1":
            self.s3.create_bucket(Bucket=self.bucket_name)
        else:
            self.s3.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.region}
                )
    
    def upload_data_to_s3(self):
        """
        Now that your Amazon S3 bucket has been created, upload the CSV file of
        our user-item-interaction data.
        """
        boto3.Session().resource('s3').Bucket(self.bucket_name).Object(self.target_file_name).upload_file(self.source_data_path)
        s3DataPath = "s3://"+self.bucket_name+"/"+self.target_file_name
    
    def set_s3_bucket_policy(self, policy=None):
        """
        Amazon Personalize needs to be able to read the contents of your S3
        bucket. So add a bucket policy which allows that.
        """
        if not policy:
            policy = {
                "Version": "2012-10-17",
                "Id": "PersonalizeS3BucketAccessPolicy",
                "Statement": [
                    {
                        "Sid": "PersonalizeS3BucketAccessPolicy",
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "personalize.amazonaws.com"
                        },
                        "Action": [
                            "s3:*Object",
                            "s3:ListBucket"
                        ],
                        "Resource": [
                            "arn:aws:s3:::{}".format(self.bucket_name),
                            "arn:aws:s3:::{}/*".format(self.bucket_name)
                        ]
                    }
                ]
            }

        self.s3.put_bucket_policy(Bucket=self.bucket_name, Policy=json.dumps(policy))

    def create_iam_role(self, role_name=None):
        """
        Amazon Personalize needs the ability to assume roles in AWS in order to
        have the permissions to execute certain tasks. Let's create an IAM role
        and attach the required policies to it. The code below attaches very permissive
        policies; please use more restrictive policies for any production application.
        """
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                "Effect": "Allow",
                "Principal": {
                    "Service": "personalize.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
                }
            ]
        }
        create_role_response = self.iam.create_role(
            RoleName = role_name,
            AssumeRolePolicyDocument = json.dumps(assume_role_policy_document)
        )

        # AmazonPersonalizeFullAccess provides access to any S3 bucket with a name that includes "personalize" or "Personalize" 
        # if you would like to use a bucket with a different name, please consider creating and attaching a new policy
        # that provides read access to your bucket or attaching the AmazonS3ReadOnlyAccess policy to the role
        policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonPersonalizeFullAccess"
        self.iam.attach_role_policy(
            RoleName = role_name,
            PolicyArn = policy_arn
        )
        # Now add S3 support
        self.iam.attach_role_policy(
            PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess',
            RoleName=role_name
        )
        time.sleep(60) # wait for a minute to allow IAM role policy attachment to propagate
        self.role_arn = create_role_response["Role"]["Arn"]

    def import_data_from_s3(self, import_job_name=None):
        """
        Earlier you created the dataset group and dataset to house your information,
        so now you will execute an import job that will load the data from the S3
        bucket into the Amazon Personalize dataset.
        """
        create_dataset_import_job_response = self.personalize.create_dataset_import_job(
        jobName = import_job_name,
        datasetArn = self.dataset_arn,
        dataSource = {
            "dataLocation": "s3://{}/{}".format(self.bucket_name, self.target_file_name)
        },
        roleArn = self.role_arn
        )
        self.dataset_import_job_arn = create_dataset_import_job_response['datasetImportJobArn']

        """
        Before we can use the dataset, the import job must be active. Execute the
        cell below and wait for it to show the ACTIVE status. It checks the status
        of the import job every minute, up to a maximum of 6 hours.
        Importing the data can take some time, depending on the size of the dataset.
        In this workshop, the data import job should take around 15 minutes.
        """
        max_time = time.time() + 6*60*60 # 6 hours
        while time.time() < max_time:
            describe_dataset_import_job_response = personalize.describe_dataset_import_job(
                datasetImportJobArn = dataset_import_job_arn
            )
            status = self.check_import_job_status()
            print("DatasetImportJob: {}".format(status))
            if status == "ACTIVE" or status == "CREATE FAILED":
                break
            time.sleep(60)
    
    def check_import_job_status(self):
        describe_dataset_import_job_response = self.personalize.describe_dataset_import_job(
            datasetImportJobArn = self.dataset_import_job_arn
        )
        status = describe_dataset_import_job_response["datasetImportJob"]['status']
        return status

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['personalize']
        del attributes['personalize_runtime']
        del attributes['s3']
        del attributes['iam']
        return attributes