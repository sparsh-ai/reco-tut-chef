import boto3
import json
import time


class personalize_model:
    def __init__(self,
                 dataset_group_arn = None,
                 solution_arn = None,
                 solution_version_arn = None,
                 campaign_arn = None,
                 filter_arns = [],
                 ):
        self.personalize = None
        self.personalize_runtime = None
        self.personalize_events = None
        self.dataset_group_arn = dataset_group_arn
        self.solution_arn = solution_arn
        self.solution_version_arn = solution_version_arn
        self.campaign_arn = campaign_arn
        self.filter_arns = filter_arns

    def setup_connection(self):
        try:
            self.personalize = boto3.client('personalize')
            self.personalize_runtime = boto3.client('personalize-runtime')
            self.personalize_events = boto3.client(service_name='personalize-events')
            print("SUCCESS | We can communicate with Personalize!")
        except:
            print("ERROR | Connection can't be established!")

    def recipe_list(self, getdict=False):
        recipes_dict = {d['name']:d['recipeArn'] for d in self.personalize.list_recipes()['recipes']}
        if getdict:
            return recipes_dict
        return list(recipes_dict.keys())

    def create_solution(self, name=None, arnname=None, hpo=False, config=None):
        """
        First you create a solution using the recipe. Although you provide
        the dataset ARN in this step, the model is not yet trained. See this as
        an identifier instead of a trained model.
        """
        if not arnname:
            arnname = name
        recipes_dict = self.recipe_list(getdict=True)
        assert arnname in list(recipes_dict.keys()), "This recipe is not available"
        if config:
            solution_response = self.personalize.create_solution(name = name,
                datasetGroupArn = self.dataset_group_arn,
                recipeArn = recipes_dict[arnname],
                solutionConfig = config
                )
        else:
            solution_response = self.personalize.create_solution(name = name,
                    datasetGroupArn = self.dataset_group_arn,
                    recipeArn = recipes_dict[arnname],
                    performHPO = hpo
                    )
        self.solution_arn = solution_response['solutionArn']
    
    def create_solution_version(self):
        solution_version_response = self.personalize.create_solution_version(
            solutionArn = self.solution_arn
            )
        self.solution_version_arn = solution_version_response['solutionVersionArn']

    def check_solution_version_status(self):
        describe_solution_version_response = self.personalize.describe_solution_version(
            solutionVersionArn = self.solution_version_arn
            )
        status = describe_solution_version_response["solutionVersion"]["status"]
        return status

    def get_evaluation_metrics(self):
        solution_metrics_response = self.personalize.get_solution_metrics(
            solutionVersionArn = self.solution_version_arn
            )
        return solution_metrics_response
    
    def create_campaign(self, name=None, min_tps=1):
        create_campaign_response = self.personalize.create_campaign(
            name = name,
            solutionVersionArn = self.solution_version_arn,
            minProvisionedTPS = min_tps
            )
        self.campaign_arn = create_campaign_response['campaignArn']
    
    def check_campaign_creation_status(self):
        version_response = self.personalize.describe_campaign(
            campaignArn = self.campaign_arn
        )
        status = version_response["campaign"]["status"]
        return status

    def create_filter(self, name=None, expression=None):
        create_filter_response = self.personalize.create_filter(name = name,
            datasetGroupArn = self.dataset_group_arn,
            filterExpression = expression
            )
        filter_arn = create_filter_response['filterArn']
        self.filter_arns.append(filter_arn)

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['personalize']
        del attributes['personalize_runtime']
        del attributes['personalize_events']
        return attributes