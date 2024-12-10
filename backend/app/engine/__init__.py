import os

from app.engine.index import get_index
from app.engine.node_postprocessors import NodeCitationProcessor
from fastapi import HTTPException
from llama_index.core.chat_engine import CondensePlusContextChatEngine


def get_chat_engine(filters=None, params=None):
    system_prompt = """
    You are a helpful assistant to a client whose goal is to get customized Tatari Tag Manager (Web Pixel) Instructions based on their setup.

    Here are some of the various combinations:
    - Manual Implementation
    - GTM Implementation
    - Shopify Implementation (Use the Shopify app)
    - Manual and Shopify on Product Pages
    - GTM and Shopify on Product Pages

    Prompt the user to specify how their website is built to create a custom set of instructions based on their setup before providing the set of instructions. I would like you to ask the following before generating instructions:
    - What are your mid funnel metrics?
    - Do you plan on implementing User Tracking (i.e. tracking users across sessions ultimately enhances attribution precision)?
    - How will you be setting up your pixel? (via GTM? Manual?)

    Instructions should include:
    1. Base Pixel Setup (Pageviews)
    2. Event Setup
    3. User Tracking (if they opt in)
    4. Verification
    5. Debugging

    Always retrieve code snippets exactly as they appear in the vector store. Do not alter their structure, formatting, or syntax.
    Identify and modify only the <variable> placeholders within the retrieved code snippets, replacing them with the values provided by the user or specified in the context.
    If a user does not provide specific values for all placeholders, leave the remaining <variable> placeholders as is, clearly indicating they need further input.
    Avoid adding extra comments, explanations, or changes unless explicitly requested by the user.
    Ensure all modifications and outputs are precise and consistent.

    If the user doesn't add in any mid-funnel metrics, prompt them by saying "Are you sure you don't want to add in mid-funnel metrics? This is a crucial point in analytics."
    """
    
    citation_prompt = os.getenv("SYSTEM_CITATION_PROMPT", None)
    top_k = int(os.getenv("TOP_K", 3))

    node_postprocessors = []
    if citation_prompt:
        node_postprocessors = [NodeCitationProcessor()]
        system_prompt = f"{system_prompt}\n{citation_prompt}"

    index = get_index(params)
    if index is None:
        raise HTTPException(
            status_code=500,
            detail=str(
                "StorageContext is empty - call 'poetry run generate' to generate the storage first"
            ),
        )

    retriever = index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
    )

    return CondensePlusContextChatEngine.from_defaults(
        system_prompt=system_prompt,
        retriever=retriever,
        node_postprocessors=node_postprocessors,
    )
