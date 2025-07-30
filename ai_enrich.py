import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
PROMPT_TEMPLATE = """
You are a business domain expert.

Your task is to read the given transcript and do the following:

1. Classify the content into exactly one of the following business categories:
   - ERP: Enterprise Resource Planning (e.g., finance, accounting, business processes)
   - HCM: Human Capital Management (e.g., HR, employee management, workforce)
   - SCM: Supply Chain Management (e.g., logistics, inventory, procurement)
   - CX: Customer Experience (e.g., customer service, sales, marketing)

2. Generate a concise and informative:
   - "title": A clear and descriptive title for the content.
   - "summary": A brief summary capturing the main idea and business value.

Respond in this exact JSON format only (no preamble, explanation, or Markdown formatting):

{
  "category": "ERP" | "HCM" | "SCM" | "CX",
  "title": "<Generated title>",
  "summary": "<Generated summary>"
}
Respond with pure JSON only — do NOT wrap the output in triple backticks or markdown formatting.
"""


def call_groq_llm(transcript_text):
    client = Groq()
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": PROMPT_TEMPLATE.strip()},
            {"role": "user", "content": transcript_text.strip()}
        ],
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    full_response = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content

    try:
        return json.loads(full_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw response:\n{full_response}")



def enrich_with_groq(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    for item in transcripts:
        transcript_text = item.get("transcript", "")
        try:
            result = call_groq_llm(transcript_text)

            # print("*************========")
            # print(result)
            # print("*******************|||")
            item["category"] = result["category"]
            item["title"] = result["title"]
            item["summary"] = result["summary"]
        except Exception as e:
            print(f"Error processing transcript: {e}")
            item["category"] = "Error"
            item["title"] = "Error"
            item["summary"] = str(e)

    output_path = file_path.replace(".json", "_enriched_groq.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2)

    # print(f"Enriched data saved to: {output_path}")

def test_ai():
    client = Groq()
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": PROMPT_TEMPLATE.strip()},
            {"role": "user", "content": """I am medium Chandni here from the Oracle CSS team and I am here to present the AI agent that my team and I have developed my team members names on your listed in the slide for your short reference and our agent is the procurement smart assistant which focuses mainly on the procurement module so to speak a more about it and diving in basically an all in one AI agent in the procurement module it takes in any data for the purchase request suppliers data which basically make the most of the procurement module and it takes in you know it takes in the information that we require and it gives us the results write it down it takes in the question that we are going to ask about a certain activity or what is the information that we require from it goes in and checks in the logic and it brings out the data that we needed
this is required for us to make some decisions to avoid avoids us in certain risks and it helps us even giving us real time data and whatever questions that we ask it gives us a possible answer out of it and it predicts the it takes in the older fast information and it also gives us what could be the future as well so and it leads to true or false
status of the PO and it will show you what you know what is the approval status and basically the overall status the smart approval mail draught is for its it allows you to draught you know you give certain information such as a PO number or supplier name and you will ask for the agent to draught a short email with the short with keywords that you would like to you know request the particular information from and you can use this draught and send it for your send it at your discretion the product you approval turn around and the tentative for rejection risk predictor these are predictor tools so the approval turn around us to it will take a past consideration of an approval how long it has taken under that particular so it will give you that prediction for future one class 12 and for non PO invoice detection this will list out all invoices that have missed
reference numbers so this is important right for audit audit during audit so this list out all the invoices so we take necessary action been from the supplier to corrected next is duplicate supply prevention this is when we have we have many suppliers registering with the same data or the same you know main parameters so it flags it and helps us to clean up this you know supplier database and only allows without any page without any duplication so detecting pure delay is when any pores any pose that I have reached the expected delivery date and it will help us to help the buyer to inform the supplier after supply problem to write an email to them to ask what's the delay in this you know requisition recommendation is for you know it's a smart buying where you can you check what you have purchased in the past how many times
particular item is there an item that you are frequently been purchasing against the customer so that is against the supplier so that gives you a prediction that gives you know what your past history has been and it can predict what you can you know by in future the new supplier performance monitoring will taken all your supplier pass data where is the delivery dates or their how their payments in every thing and it helps us to with certain Matrix it helps us to rate the suppliers performance and probably if you would like to continue with them or he is not been a good supply you can take you can take a call on that agent as like a non pure invoice detection it to find out if the if the buyer name is not injected into the PR so this is also for auditing purposes you can use Predator is when you have any you know we can avoid certain of the PO getting rejected in the past date of why
could have got rejected those reasons can be listed out and you can take the necessary action as a buyer to avoid this reason future so this is what are agent is about and what are agent uses is as like it takes in your questions at the source is going to go and check the data of what logic with built in the custom logic and what tools we have built in the agent it uses them and it brings out the answers for you there is no you know there is no false data it gives you what data is actually deciding the system what you accomplish obviously as we very well know there is less time and the results that we are going to get and the number of clicks is going to be less or do you get an analysis out of it you reduce the risk you mitigate the risks and also you get a healthier you know supply base where you are what duplication and your turn around time and everything will be much more better so jumping into the demo of this this is my agent
sorry this is a smart procurement assistant I have the main supervisor agent and all the sub worker worker agents under it so I am going ahead and run the agent ok so you see these are the start a questions I am going to hit the first one so I need some information on my pose right so I'm just gonna go ahead dinner as you can see it is asking for a period so I am asking what is the status of my jio is going to take the corresponding to the corresponding
in this case it's going to go to the life cycle agent the pure life cycle agent to take the status of your pure so it is giving you the current status so now I am going to ask for the same why is this is this PO is this pure delivered this is again because this is not the right question just give me a minute it's taking again the right you know it's taking the same context but let me ask with the questions a bit but in fact checking the schedules so it's checking for the expected delivery date and it saying that is not there in the correct result as expected so it is not a delivered so now what you can do
you can write an email so draught an email to the supplier to have is delivered so as you can see I am using the same prompt to have multiple things done at the same time so we are drafting an email not so this is another agent draught of what you can take it up and you can paste it in your email and you can send it across to your supplier now I am going to ask how has this supply performed so this is another worker told so
what you know what it takes into consideration so as you can see it's taking another supplier performance monitoring agent here and it is considering all these tools such as the Pose it touches the invoices and it takes on all the data and is going to give us a result please play with me that has given us the result that we needed so this is this is a few task demo that I have done and you know we have much more here in place in this way we have all these tools working with different tasks so according to what I ask is the agent is going to give us the data space we do have certain you know action task on them and that is in the next phase which we are already
development phase so please look forward to it I hope you like the agent that we have built in case you do please go ahead and vote for this agent and help us make it better thank you for being here and listening to me I hope this this is helpful for you as well thank you and have a good day"""}
        ],
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")
