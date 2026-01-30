import os
import json
import time
import traceback
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_gigachat import GigaChat
from erc3 import store, ApiException, TaskInfo, ERC3


def error_json(message: str) -> str:
    """Return error as JSON string (required for GigaChat compatibility)."""
    return json.dumps({"error": message})

# pip install python-dotenv
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def create_llm(model: str, provider: str = "openrouter"):
    """Create LLM instance based on provider.
    
    Args:
        model: Model name/ID
        provider: 'openrouter' or 'gigachat'
    
    Returns:
        LLM instance
    """
    if provider == "gigachat":
        return GigaChat(
            model=model,
            verify_ssl_certs=False,
            profanity_check=False,
            timeout=120,
        )
    else:  # openrouter
        return ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=4096,
        )

SYSTEM_PROMPT = """
You are a business assistant helping customers of OnlineStore. Your goal is to complete purchases at the LOWEST possible price.

CRITICAL RULES:

1. PRICE OPTIMIZATION (when task mentions "cheap", "discount", or multiple coupons):
   - You MUST test ALL combinations and ALL coupons to find the true minimum
   - Coupons may give DIFFERENT discounts for DIFFERENT product combinations!
   - A coupon showing discount=0 means it doesn't apply to that basket - TRY OTHER COMBINATIONS
   
2. STEP-BY-STEP OPTIMIZATION PROCESS:
   a) List ALL relevant products with their prices and pack sizes
   b) Generate ALL valid combinations that satisfy the quantity requirement
   c) For EACH combination:
      - Clear basket (remove all items)
      - Add that combination's products
      - Try EACH coupon and use view_basket to see actual discount
      - Record: combination + coupon + total price
   d) Compare ALL recorded totals
   e) Checkout with the CHEAPEST combination + coupon

3. EXAMPLE: "Buy 24 sodas" with 6pk ($12), 12pk ($20), 24pk ($35):
   - Option A: 1×24pk = $35
   - Option B: 2×12pk = $40
   - Option C: 4×6pk = $48
   - Option D: 1×12pk + 2×6pk = $20 + $24 = $44
   - Option E: 1×6pk + 1×12pk + 6 = doesn't work
   Then try EACH coupon with EACH option to find which gives lowest total!

4. Pagination: If NextOffset > 0, continue listing products. If NextOffset = -1, no more products.

5. Use SKU field when adding products to basket.

6. CLEARING BASKET: To test different combinations:
   - First use view_basket to see current items and their quantities
   - Remove each item using remove_item_from_basket with the EXACT quantity shown in basket
   - quantity must be positive (e.g., if basket has 3 of "soda-6pk", use quantity=3)

7. After finding cheapest option, clear basket, add winning combination, apply best coupon, and CHECKOUT.

8. CRITICAL: NEVER ask questions or wait for confirmation. ALWAYS complete the task autonomously.
   - DO NOT say "Should I proceed?" - just proceed!
   - DO NOT stop without completing checkout
   - If something fails, try a different approach and continue
   - Your job is done ONLY when checkout_basket returns success
"""

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_YELLOW = "\x1B[33m"
CLI_CLR = "\x1B[0m"


def create_store_tools(store_api):
    """Create LangChain tools for store operations."""
    
    @tool
    def list_products(offset: int = 0, limit: int = 10) -> str:
        """List products from the store catalog with pagination.
        
        Args:
            offset: Pagination offset (use NextOffset from previous response to get more results)
            limit: Maximum number of products to return (default 10, max 50)
        
        Returns:
            JSON with Products array and NextOffset (if more products available)
        """
        try:
            req = store.Req_ListProducts(offset=offset, limit=limit)
            result = store_api.dispatch(req)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}list_products(offset={offset}, limit={limit}){CLI_CLR}")
            print(f"  -> {txt[:300]}{'...' if len(txt) > 300 else ''}")
            return txt
        except ApiException as e:
            print(f"{CLI_RED}list_products ERR: {e.detail}{CLI_CLR}")
            return error_json(e.detail)
        except Exception as e:
            print(f"{CLI_RED}list_products EXCEPTION: {e}{CLI_CLR}")
            traceback.print_exc()
            return error_json(str(e))
    
    @tool
    def view_basket() -> str:
        """View current shopping basket contents, applied discounts and total.
        
        Returns:
            JSON with basket Items, Coupon info, Discount and Total price
        """
        try:
            req = store.Req_ViewBasket()
            result = store_api.dispatch(req)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}view_basket(){CLI_CLR}")
            print(f"  -> {txt}")
            return txt
        except ApiException as e:
            print(f"{CLI_RED}view_basket ERR: {e.detail}{CLI_CLR}")
            return error_json(e.detail)
        except Exception as e:
            print(f"{CLI_RED}view_basket EXCEPTION: {e}{CLI_CLR}")
            traceback.print_exc()
            return error_json(str(e))
    
    @tool
    def add_product_to_basket(sku: str, quantity: int) -> str:
        """Add a product to the shopping basket by SKU.
        
        Args:
            sku: The SKU (stock keeping unit) of the product to add - get it from list_products
            quantity: Number of items to add (must be positive)
        
        Returns:
            Confirmation or error message
        """
        try:
            req = store.Req_AddProductToBasket(sku=sku, quantity=quantity)
            result = store_api.dispatch(req)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}add_product_to_basket(sku={sku}, quantity={quantity}){CLI_CLR}")
            print(f"  -> {txt}")
            return txt
        except ApiException as e:
            print(f"{CLI_RED}add_product_to_basket ERR: {e.detail}{CLI_CLR}")
            return error_json(e.detail)
        except Exception as e:
            print(f"{CLI_RED}add_product_to_basket EXCEPTION: {e}{CLI_CLR}")
            traceback.print_exc()
            return error_json(str(e))
    
    @tool
    def remove_item_from_basket(sku: str, quantity: int) -> str:
        """Remove a product from the shopping basket.
        
        Args:
            sku: The SKU of the product to remove
            quantity: Number of items to remove (must be positive, e.g., to remove all 3 items pass quantity=3)
        
        Returns:
            Confirmation or error message
        """
        try:
            req = store.Req_RemoveItemFromBasket(sku=sku, quantity=quantity)
            result = store_api.dispatch(req)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}remove_item_from_basket(sku={sku}, quantity={quantity}){CLI_CLR}")
            print(f"  -> {txt}")
            return txt
        except ApiException as e:
            print(f"{CLI_RED}remove_item_from_basket ERR: {e.detail}{CLI_CLR}")
            return error_json(e.detail)
        except Exception as e:
            print(f"{CLI_RED}remove_item_from_basket EXCEPTION: {e}{CLI_CLR}")
            traceback.print_exc()
            return error_json(str(e))
    
    @tool
    def apply_coupon(coupon: str) -> str:
        """Apply a coupon code to get discounts. Only one coupon can be active at a time.
        
        Args:
            coupon: The coupon code to apply
        
        Returns:
            Confirmation with discount info or error message
        """
        try:
            req = store.Req_ApplyCoupon(coupon=coupon)
            result = store_api.dispatch(req)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}apply_coupon(coupon={coupon}){CLI_CLR}")
            print(f"  -> {txt}")
            return txt
        except ApiException as e:
            print(f"{CLI_RED}apply_coupon ERR: {e.detail}{CLI_CLR}")
            return error_json(e.detail)
        except Exception as e:
            print(f"{CLI_RED}apply_coupon EXCEPTION: {e}{CLI_CLR}")
            traceback.print_exc()
            return error_json(str(e))
    
    @tool
    def remove_coupon() -> str:
        """Remove the currently applied coupon from the basket.
        
        Returns:
            Confirmation or error message
        """
        try:
            req = store.Req_RemoveCoupon()
            result = store_api.dispatch(req)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}remove_coupon(){CLI_CLR}")
            print(f"  -> {txt}")
            return txt
        except ApiException as e:
            print(f"{CLI_RED}remove_coupon ERR: {e.detail}{CLI_CLR}")
            return error_json(e.detail)
        except Exception as e:
            print(f"{CLI_RED}remove_coupon EXCEPTION: {e}{CLI_CLR}")
            traceback.print_exc()
            return error_json(str(e))
    
    @tool
    def checkout_basket() -> str:
        """Complete the purchase and checkout the basket. Call this when all items are added.
        
        Returns:
            Order confirmation with order ID or error message
        """
        try:
            req = store.Req_CheckoutBasket()
            result = store_api.dispatch(req)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}checkout_basket(){CLI_CLR}")
            print(f"  -> {txt}")
            return txt
        except ApiException as e:
            print(f"{CLI_RED}checkout_basket ERR: {e.detail}{CLI_CLR}")
            return error_json(e.detail)
        except Exception as e:
            print(f"{CLI_RED}checkout_basket EXCEPTION: {e}{CLI_CLR}")
            traceback.print_exc()
            return error_json(str(e))
    
    return [
        list_products,
        view_basket,
        add_product_to_basket,
        remove_item_from_basket,
        apply_coupon,
        remove_coupon,
        checkout_basket,
    ]


def run_agent(model: str, api: ERC3, task: TaskInfo, provider: str = "openrouter"):
    """Run LangChain agent to complete a store task.
    
    Args:
        model: Model name/ID
        api: ERC3 API client
        task: Task to complete
        provider: LLM provider - 'openrouter' or 'gigachat'
    """
    
    store_api = api.get_store_client(task)
    tools = create_store_tools(store_api)
    
    # Create LLM based on provider
    llm = create_llm(model, provider)
    
    # Create the agent using LangChain's create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    
    print(f"{CLI_YELLOW}Starting agent for task:{CLI_CLR} {task.task_text}")
    started = time.time()
    
    # Run the agent
    result = agent.invoke({
        "messages": [{"role": "user", "content": task.task_text}]
    })
    
    duration = time.time() - started
    
    # Extract final message and collect usage statistics
    final_message = result["messages"][-1]
    
    # Aggregate token usage from all AI messages
    total_prompt = 0
    total_completion = 0
    for msg in result["messages"]:
        if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
            total_prompt += msg.usage_metadata.get('input_tokens', 0)
            total_completion += msg.usage_metadata.get('output_tokens', 0)
    
    print(f"\n{CLI_GREEN}Agent completed in {duration:.2f}s{CLI_CLR}")
    print(f"Tokens: {total_prompt} prompt + {total_completion} completion = {total_prompt + total_completion} total")
    print(f"Final response: {final_message.content}")
    
    # Log LLM usage
    api.log_llm(
        task_id=task.task_id,
        completion=final_message.content,
        model=model,
        duration_sec=duration,
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
    )
