from google import genai

client = genai.Client(api_key="AIzaSyCTmf2trLBuQqqLwMacvI3hJ0AHUj6zkdc")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="""
    Analyze the following real estate property data. Based on the string fields like title, description, and propertyType, determine if this is likely a legitimate real estate listing or spam or fake listing or ot of contexte stuff. Return only true if it seems real, or false if it seems like spam.
  Return only true or false
the data : 
title: "house for sale"
description: "beautifull 3 bedroom apartement"
propertyType: "Sale"
    
    """
)
print(response.text)