from datamodel import GarmentDataModel
import config as cfg

model =
query = 'SELECT * FROM clothingimagetag LIMIT 10'
data = model.query(query)

model.close_connection()
