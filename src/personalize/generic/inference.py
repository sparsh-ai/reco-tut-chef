import boto3
import json
import time


class personalize_inference:
    def __init__(self,
                 dataset_group_arn = None,
                 campaign_arn = None,
                 event_tracker_arn = None,
                 role_arn=None,
                 solution_version_arn=None,
                 batch_job_arn=None
                 ):
        self.personalize = None
        self.personalize_runtime = None
        self.personalize_events = None
        self.dataset_group_arn = dataset_group_arn
        self.campaign_arn = campaign_arn
        self.event_tracker_arn = event_tracker_arn
        self.event_tracker_id = event_tracker_id
        self.role_arn = role_arn
        self.solution_version_arn = solution_version_arn
        self.batch_job_arn = batch_job_arn

    def setup_connection(self):
        try:
            self.personalize = boto3.client('personalize')
            self.personalize_runtime = boto3.client('personalize-runtime')
            self.personalize_events = boto3.client(service_name='personalize-events')
            print("SUCCESS | We can communicate with Personalize!")
        except:
            print("ERROR | Connection can't be established!")

    def get_recommendations(self, itemid=None, userid=None, k=5,
                            filter_arn=None, filter_values=None):
        get_recommendations_response = self.personalize_runtime.get_recommendations(
            campaignArn = self.campaign_arn,
            itemId = str(itemid),
            userId = str(userid),
            filterArn = filter_arn,
            filterValues = filter_values,
            numResults = k
            )
        
    def get_rankings(self, userid=None, inputlist=None):
        get_recommendations_response = self.personalize_runtime.get_personalized_ranking(
            campaignArn = self.campaign_arn,
            userId = str(userid),
            inputList = inputlist
            )
        
    def create_event_tracker(self, name=None):
        response = self.personalize.create_event_tracker(
            name=name,
            datasetGroupArn=self.dataset_group_arn
            )
        self.event_tracker_arn = response['eventTrackerArn']
        self.event_tracker_id = response['trackingId']
    
    def put_events(self, userid=None, sessionid=None, eventlist=None):
        self.personalize_events.put_events(
            trackingId = self.event_tracker_id,
            userId = userid, 
            sessionId = sessionid,
            eventList = eventlist
            )
        
    def create_batch_job(self, jobname=None, input_path=None, output_path=None):
        response = self.personalize.create_batch_inference_job(
            solutionVersionArn = self.solution_version_arn,
            jobName = jobname,
            roleArn = self.role_arn,
            jobInput = {"s3DataSource": {"path": input_path}},
            jobOutput = {"s3DataDestination": {"path": output_path}}
            )
        self.batch_job_arn = response['batchInferenceJobArn']
    
    def check_batch_job_status(self):
        batch_job_response = self.personalize.describe_batch_inference_job(
        batchInferenceJobArn = self.batch_job_arn
        )
        status = batch_job_response["batchInferenceJob"]['status']
        return status

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['personalize']
        del attributes['personalize_runtime']
        del attributes['personalize_events']
        return attributes