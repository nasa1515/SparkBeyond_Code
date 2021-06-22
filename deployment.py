from sparkbeyond._api2 import classes as sb
from sparkbeyond.predictionserver.api import PredictionServerClient
import pandas as pd
import time


#discovery server option
server_address = 'http://20.194.19.101'
api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJhOGU4Y2I5MS00ZmYwLTRkODktYWMzNy0xYTFkYjE0ODEyMDYifQ.eyJpYXQiOjE2MjM4MjQ1NjcsImp0aSI6IjQ0ZTUyMmUyLWYzMDItNGYzNy05MzE3LWRjZTZhZDEzZTdhYSIsImlzcyI6Imh0dHA6Ly8yMC4xOTQuMTkuMTAxL2F1dGgvcmVhbG1zL3NwYXJrYmV5b25kIiwiYXVkIjoiaHR0cDovLzIwLjE5NC4xOS4xMDEvYXV0aC9yZWFsbXMvc3BhcmtiZXlvbmQiLCJzdWIiOiIxMjFiZTE0Ni00ZjYzLTRmMzMtYjE3NC1kZTljODBlZTRiMzMiLCJ0eXAiOiJPZmZsaW5lIiwiYXpwIjoiZGlzY292ZXJ5LXNkayIsInNlc3Npb25fc3RhdGUiOiI1NzYyZTIyNC1lZjA4LTQ4ODktOTYxYy1kY2JkMjY0NDg2MmMiLCJzY29wZSI6InByb2ZpbGUgZW1haWwgb2ZmbGluZV9hY2Nlc3MifQ.3EQl2Qm2M1aM5RlQnR9n1g9fTCT7W7jQn1fxHkwIUro'

# prediction server option
server_address2 = 'http://52.141.2.39'
api_token = 'eyJhbGciOiJIUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI4NDIwYjBiOS05MzgwLTQ1ZmQtYmRhMy1jZjk3YjczYmUwZWYifQ.eyJpYXQiOjE2MjM4MjI0MzksImp0aSI6IjA1MzllMmM2LTIzNjMtNDg1YS1hNzIyLWMwZDVhMGU5NDYxNyIsImlzcyI6Imh0dHA6Ly81Mi4xNDEuMi4zOS9hdXRoL3JlYWxtcy9zcGFya2JleW9uZCIsImF1ZCI6Imh0dHA6Ly81Mi4xNDEuMi4zOS9hdXRoL3JlYWxtcy9zcGFya2JleW9uZCIsInN1YiI6Ijc1YTkxMTg5LTY3ZGQtNDZjYy1hMTUxLWE0ODFiMzU4OWFlOSIsInR5cCI6Ik9mZmxpbmUiLCJhenAiOiJlcHMtY29uc3VtZXItYXBwIiwic2Vzc2lvbl9zdGF0ZSI6ImM2NzNkOTQ4LTZhZTQtNGVlMC04YzY4LTk0NmM4NmI0YzNlYyIsInNjb3BlIjoicHJvZmlsZSBlbWFpbCBvZmZsaW5lX2FjY2VzcyJ9.fIC9UrSzt3ZWB9BIA_8aPugWky2Ts0_LTiNSDfSk7Bg'


# model download option
train_file = '/home/data/train.csv'
project = 'titanic_survived'
revision = '2'
zipname = 'bae.zip'
download_path = f'/home/zip/{zipname}'
group = 'baegroup'


# Predict : data file path option
data_filename = 'test.csv'
file_path = f'/home/data/{data_filename}'
name = 'titanic.csv.gz'
resultname = '2021-06-21-Result'


####    ########
####    ########
#Crate Model Code
####    ########
####    ########

# load SB Discovery Client
client = sb.SparkBeyondClient(server_address, api_key)
train_df = pd.read_csv(train_file)
titanicSurvivedModel = client.learn(
                                    project_name=project,  # your choice of project name
                                    revision_description = "Titanic naive",
                                    train_data = train_df, #Since a test set is not provided separately, your main that will be split into Train and Test (80%, 20% respectively)
                                    target = "survived" # your target
                                    )

model = client.revision(project,revision)
model.download_model(download_path, with_contexts = True)


print("creating model....")
time.sleep(5)


####    ########
####    ########
# Download Model Code
####    ########
####    ########

client = PredictionServerClient(url=server_address2, refresh_token=api_token)
client.upload_group(zip_file_path=download_path ,group_name=group ,force=True)

for group in client.get_groups():
        print(group.name)



print("uploading Model....")
time.sleep(5)
####    ########
####    ########
# Predict code
####    ########
####    ########

# dataset create
test_df = pd.read_csv(file_path)
predict_result = client.predict(input_df=test_df, model=group, include_enriched=True, include_prediction=True, include_class_ranks_as_map=True, dest_filename=resultname)


print("sleep 10 seconds")
time.sleep(10)


result_df = pd.read_csv(resultname)
print(result_df)

