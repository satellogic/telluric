import os
from tile_server import TileServer
# azure_account_name = "dsssatellogic"
# azure_account_key = "Yp1CQV3p0wXsLUy0NtIf4c16ZZvTT/SYgDVYL/xGKhLW7xBvji3uLPcusJhBwpABDRSFXobkPwnctzSXhJ80Fg=="

azure_account_name = "ariel"
azure_account_key = "LUbIzVc31UkjxqRHFcY/BSQxpOvFKvUmEMys6M1jsE73JHjPc/6/cha5VfGmVeMAyg5KUbyL/c0dVUaaqoeVEA=="

os.environ["AZURE_ACCOUNT_NAME"] = azure_account_name
os.environ["AZURE_ACCOUNT_KEY"] = azure_account_key

# feature_collections = os.environ.get("TELLURIC_FEATURE_COLLECTIOS", "./miniserver_demo/ds2_remote.json")
feature_collections = os.environ.get("TELLURIC_FEATURE_COLLECTIOS", "./ds2.json")
feature_collections = feature_collections.split(',')

ts = TileServer(feature_collections)

print(ts.get_start_point())
print(ts.get_folium_client())
ts.run()
