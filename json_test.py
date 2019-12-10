import json

with open('regression_XL_configuration.json', "r", encoding='utf-8') as reader:
  json_config = json.loads(reader.read())
  for key, value in json_config.items():
    print(key,value)