# el maper, que transforma el json de mongodb a un json con el formato de nuestra entidad, no realiza la transformacion a nuestra entidad,
# solo el formato del json

def document_schema(document) -> dict:
    return {
        "id": str(document["_id"]),
        "name": document["name"],
        "name_document": document["name_document"],
        "date": document["date"] 
    }


 
def documents_schema(users) -> list:
    return [ document_schema( user ) for user in users]