from whatstk import df_from_whatsapp
from whatstk.data import whatsapp_urls

df = df_from_whatsapp(whatsapp_urls.POKEMON)
records = df.to_dict(orient="records")

# TODO:
# READ IN export.zip
# Process text
# Process audio (transcribe mp3 to text)
# Return Records


print(records)