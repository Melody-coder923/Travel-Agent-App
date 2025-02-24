from textwrap import dedent
from agno.agent import Agent,RunResponse
from agno.tools.serpapi import SerpApiTools
import streamlit as st
from agno.models.groq import Groq
from agno.tools.dalle import DalleTools
from agno.utils.log import logger

from datetime import datetime
import json
from rich.pretty import pprint
from typing import List
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agno.tools.file import FileTools
from pathlib import Path

import io  # Import the io module
import base64  # Import the base64 module
import os,sys


# Set up the Streamlit app
st.title("AI Travel Planner ✈️")
st.caption("Plan your next adventure with AI Travel Planner by researching and planning a personalized itinerary on autopilot using llama")

# Get Groq API key from user
groq_api_key = "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"

# Get SerpAPI key from the user
serp_api_key = "3ee1ceb9aef090663ad0ee7ebbe82b2ecc15efb82a8eebedf78a4839a68a32c1"

#Define Class
class DailyActivity(BaseModel):
    description: str = Field(
        ..., 
        description="The description of the activity for the day "
        "and includes the day number as prefix, e.g 'Day 1: Arrive'"
    )
class Itinerary(BaseModel):
    days: List[DailyActivity] = Field(
        ...,
        description="A list of activities for each day of the trip",
    )


class Cost(BaseModel):
    total_cost: int=Field(...,
        description="The overall cost of the trip, including all expenses.",
        )
    accomodation_cost: int=Field(...,
        description="The cost of lodging during the trip, such as hotels or Airbnb rentals.",
        )
    transport_cost: int=Field(...,
        description="The cost of transportation, including flights, trains, car rentals, or public transit.",
        )
    ticket_cost: int=Field(...,
        description="The cost of tickets for attractions, tours, or events during the trip.",
        )
    food_cost: int=Field(...,
        description="The cost of meals and snacks during the trip",
        )

class Output(BaseModel):
    destination: str = Field(
        ...,
        description="The main destination of the trip."
    )
    duration: int = Field(
        ...,
        description="The number of days for the trip."
    )
    cost: Cost = Field(
        ...,
        description="The cost details of the trip."
    )

    itinerary : Itinerary = Field(
        ...,
        description="The detailed itinerary of the trip",
    )
    

if groq_api_key and serp_api_key:
    researcher = Agent(
        name="Researcher",
        role="Searches for travel destinations, activities, and accommodations based on user preferences",
        model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
        description=dedent(
            """\
        You are a world-class travel researcher. Given a travel destination,the number of days, the number of travlers,and the estimated expense that the user wants to travel for,
        generate a list of search terms for finding relevant travel activities and accommodations.
        Then search the web for each term, analyze the results, and return the 10 most relevant results.
        """
        ),
        instructions=[
            "Given a travel destination,the number of days, the number of travlers,and the estimated expense that the user wants to travel for, first generate a list of 3 search terms related to that travel destination,the number of days, the number of travlers,and the estimated expense.",
            "For each search term, `search_google` and analyze the results."
            "From the results of all searches, return the 10 most relevant results to the user's preferences.",
            "Remember: the quality of the results is important.",
            
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )
    planner = Agent(
        name="Planner",
        role="Generates a draft itinerary based on user preferences and research results",
        model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
        description=dedent(
            """\
        You are a senior travel planner. travel destination,the number of days, the number of travlers,and the estimated expense, and a list of research results,
        your goal is to generate a draft itinerary that meets the user's needs and preferences.
        """
        ),
        instructions=[
            "Given a travel destination,the number of days, the number of travlers,and the estimated expense that the user wants to travel for, first generate a list of 3 search terms related to that travel destination,the number of days, the number of travlers,and the estimated expense, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
            "Ensure the itinerary is well-structured, informative, and engaging.",
            "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
            "Remember: the quality of the itinerary is important.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
            "Give structured output that fits into the following class definition"
            "Class Output has attributes destination (str), duration (int), cost(Class Cost), itinerary (Class Itinerary)"
            "Class Itinerary has attributes itinerary (a list of objects belong to Class DailyActivity)"
            "Class DailyActivity has attribute description(str)"
            "Class Cost has attributes total_cost(int), accomodation_cost(int),transport_cost(int),ticket_cost(int),food_cost(int)"
        ],
        add_datetime_to_instructions=True,
        response_model=Output,
    )
    
    
    # Input fields for the user's destination,the number of days, and estimated travle expense they want to travel for
    destination = st.text_input("Where do you want to go?")
    num_days = st.number_input("How many days do you want to travel for?", min_value=0, max_value=30)
    num_persons=st.number_input("How many persons for this trip ?",min_value=0)
    total_budget=st.number_input("How much is the estimated travel expense (US dollar) ?",min_value=0)
    if st.button("Generate Itinerary"):
        with st.spinner("Processing..."):
            # Get the response from the assistant
            response = planner.run(f"{destination} for {num_days} days for {num_persons} persons under {total_budget} US dollar", stream=False)
            # st.write(response.content)
            if isinstance(response.content, Output):
                markdown_content = f"""
# AI Travel Planner: {destination} Itinerary

This itinerary was generated by an AI Travel Planner.

## Itinerary:
"""
                output_obj = response.content
                itinerary_obj = output_obj.itinerary
                cost_obj = output_obj.cost
                for day in itinerary_obj.days:
                    st.write(day.description)
                    markdown_content += f"{day.description}\n\n"
                
                
                cost_pie_chart = [cost_obj.accomodation_cost,
                                  cost_obj.transport_cost,
                                  cost_obj.ticket_cost,
                                  cost_obj.food_cost
                                  ]
                y = np.array(cost_pie_chart)
                labels = ["Accomodation", "Transport", "Ticket", "Food"]
                fig1, ax1 = plt.subplots()
                st.divider()
                st.header("Cost Breakdowns 💰")
                markdown_content += "## Cost Breakdown:"
                colors = ["#ff69b4", "#66b3ff", "#ffff99", "#ccccff"]
                def absolute_value(val):
                    a  = np.round(val/100.*y.sum(), 0)
                    return f'${int(a)}'
                ax1.pie(y, labels=labels, autopct=absolute_value, startangle=90,colors=colors)
                # ax1.pie(y, labels=labels, autopct=lambda p: f'{p:.1f}US$', startangle=90,colors=colors)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)
                #Save figure to a buffer for markdown embedding
                buf = io.BytesIO()
                fig1.savefig(buf, format='png')
                data = base64.b64encode(buf.getbuffer()).decode("ascii")
                markdown_content += f"""
![Cost Breakdown Pie Chart](data:image/png;base64,{data})

"""
                markdown_content += f"**Total Cost: {cost_obj.total_cost}US$**\n"
                st.subheader(f"Total Cost: {cost_obj.total_cost}US$ ")
                
                # 保存到 Markdown 文件
                with open(f"{destination}_itinerary.md", "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                download_file = f"{destination}_itinerary.md"
                download_data = markdown_content
                try:
                    res = os.system(f"pandoc {destination}_itinerary.md -o {destination}_itinerary.pdf")
                    if res == 0:
                        download_file = f"{destination}_itinerary.pdf"
                except:
                    download_file = f"destination_itinerary.md"
                
                with open(download_file,"rb") as f:
                    download_data = f.read()
                st.download_button(
                    label="Download Travel Plan",
                    data= download_data,
                    file_name= download_file,
                    mime="text/pdf",
                )
                
