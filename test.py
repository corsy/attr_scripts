from datamodel import GarmentDataModel
import config as cfg

model = GarmentDataModel(cfg.mysql_hostname,
                         cfg.mysql_username,
                         cfg.mysql_userpasswd,
                         cfg.mysql_dbname)

query = 'SELECT * FROM clothingimagetag LIMIT 10'
data = model.query(query)

model.close_connection()
