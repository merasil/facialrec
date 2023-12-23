import configparser
config = configparser.ConfigParser()
config.read("config.ini")

if config["basic"]["debug"] == "True":
    print("True")
else:
    print("False")