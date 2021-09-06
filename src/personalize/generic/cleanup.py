import boto3
import json
import time


class personalize_cleanup:
    def __init__(self):
        self._setup_connection()

    def _setup_connection(self):
        try:
            self.personalize = boto3.client('personalize')
            print("SUCCESS | We can communicate with Personalize!")
        except:
            print("ERROR | Connection can't be established!")

    def delete_campaign(self, campaign_arn):
        self.personalize.delete_campaign(campaignArn = campaign_arn)
    
    def delete_solution(self, solution_arn):
        self.personalize.delete_solution(solutionArn = solution_arn)
    
    def delete_tracker(self, tracker_arn):
        self.personalize.delete_event_tracker(eventTrackerArn = tracker_arn)
    
    def delete_filter(self, filter_arn):
        self.personalize.delete_filter(filterArn = filter_arn)
    
    def delete_dataset(self, dataset_arn):
        self.personalize.delete_dataset(datasetArn = dataset_arn)

    def delete_schema(self, schema_arn):
        self.personalize.delete_schema(schemaArn = schema_arn)

    def delete_dataset_group(self, dataset_group_arn):
        self.personalize.delete_dataset_group(datasetGroupArn = dataset_group_arn)