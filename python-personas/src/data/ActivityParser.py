import re
from typing import Optional, Tuple
import pandas as pd

def get_last_part_of_url(url: str, n: int = 1) -> str:
    """Extracts the last n parts of the url, parts defined by '/'"""
    return "/".join(url.split("?")[0].split("/")[-n:])


def is_integer(s: str) -> bool:
    """Verifies whether a string is an integer"""
    try:
        int(s)
        return True
    except ValueError:
        return False


class ActivityParser:
    brand_names = [brand_name.lower() for brand_name in [
        "Ray-Ban",
        "Oakley",
        "Costa",
        "Maui Jim",
        "Celine",
        "Tom Ford",
        "Arnette",
        "Prada",
        "Miu Miu",
        "Miu",
        "Prada Linea Rossa",
        "Bvlgari",
        "Bulgari"
        "Dolce & Gabba",
        "Dolce and Gabbana",
        "Dolce",
        "Dolce &amp; Gabba",
        "Dolce &amp; Gabbana",
        "Tory Burch",
        "Persol",
        "Ralph Lauren",
        "Polo Ralph Lauren",
        "Ralph",
        "Fendi",
        "Christian Dior",
        "Dior",
        "Michael Kors",
        "Coach",
        "Tiffany & Co.",
        "Tiffany and Co.",
        "Tiffany & Co",
        "Tiffany and Co",
        "Tiffany co",
        "TiffanyCo",
        "Tiffany",
        "Versace",
        "Vogue Eyewear",
        "Vogue",
        "Carrera",
        "Burberry",
        "Gucci",
        "Chanel",
        "Armani Exchange",
        "Armani",
        "Emporio Armani",
        "Giorgio Armani",
        "Alain Mikli",
        "Valentino",
        "Sunglass Hut Collection",
        "Off White",
        "Oliver Peoples"
    ]]
    important_actions = {
        "[pdp-image-modal-close][Close]": "large image",
        "[X Pdp Prod AddToWishList][Addtofavorites]": "add to fav",
        "[X Pdp Prod AddToWishList][Addtofavourites]": "add to fav",
        "[:#][Loadmoresunglasses]": "load more",
        "[X Pdp Prod AddCart][Addtobag]": "add to cart",
        "[openProductInfoPopupBtn][ProductInformation]": "detailed product info",
        "[X CartPage Promocode Submit][Applytoorder]": "promo code",
        # "[X Pdp FindStoreOverlay Find][Find]": "Find close store",
    }
    # slide_control = re.compile(r"\[slick-slide-control\d+\]\[\d+\]")
    product_match = re.compile(r"^0?[a-z]{2} ?\d{3,}[a-z]?$", )  # matches most products like rb3025
    sunglasses_re = re.compile(r"(sun)? ?glass?(es)?")  # matches the word sunglass(es) and variations

    def __init__(self, parse_actions=False):
        self.parse_actions = parse_actions

    def parse_activity(self, row: dict, transform=True) -> Tuple[Optional[str], Optional[str]]:
        """Extracts the data from a single row to create an activity out of it. If transform is true,
        also aggregates the activity into a larger group."""
        events = row['events']
        action = row['action']

        # check if we can skip parsing
        if self.parse_actions:
            if action is not None and pd.isna(action)==False:
                index = action.find("[")
                action = action[index:]
            # if we parse actions, we need a pageview, specific action or slide control action
            if ((events is None or pd.isna(events) or "PageView" not in events)
                    and action not in self.important_actions and pd.isna(action)==False
                    and "slick-slide-control" not in action):
                return (None, None) if transform else None
        elif events is None or "PageView" not in events:
            # if we only page parseiews, we need a 'PageView' action
            return (None, None) if transform else None

        # if we are parsing because of the pageview, we ignore the action
        if not self.parse_actions or (events is not None and pd.isna(events)==False and "PageView" in events):
            action = None

        url = row["url"]
        activity = row["page_name"]

        if activity is None:
            return (None, None) if transform else None

        combo = row['combo']
        search_term = row['search_keyword']

        # # remove pre-pended ":"
        # if activity[0] == ":":
        #     activity = activity[1:]

        # modify activities that are not well-defined
        if activity == ":Plp":
            # all Plp activities with nothing more are actually search activities
            activity = ":Search"

        if activity == ":/MADISONSSTOREFRONTASSETSTORE/PAGES/CONTENT.JSP":
            activity = get_last_part_of_url(url) + ":static"
        elif activity == ":/TREND":
            activity = get_last_part_of_url(url, n=2)
        elif activity == ":Clp":
            if url[:24] == "/CoreMediaContentDisplay":
                # extract actual page
                activity = url.split("SeoSegment=")[-1].split("&")[0] + ":clp"
            else:
                activity = get_last_part_of_url(url) + ":clp"
        elif activity == ":Pdp":
            activity = get_last_part_of_url(url).split("-")[-1] + ":pdp"
        elif activity == ":Search":
            if "/SearchDisplay?" in url:
                if "%2522" in url:
                    activity = url.split("%2522")[1].replace("%2B", " ") + ":facet_page:search"
                elif "facet" in url:
                    activity = "unknown:facet_page:search"
                elif search_term is not None and pd.isna(search_term)==False and search_term != "":
                    activity = search_term + ":search"
                else:
                    activity = "unknown:search"
            elif "face" in url:
                activity = get_last_part_of_url(url) + ":plp"
            else:
                activity = get_last_part_of_url(url, n=2)
        elif activity == ":Unknown":
            if url == "/ChallengeQuestion":
                activity = "challenge_question"
            elif "logon" in url.lower():
                activity = "logon"
            elif "/UserOrderDetailsView?" in url:
                activity = "order_status:details"
            else:
                activity = get_last_part_of_url(url)
        elif activity == ":Signup":
            if "/CreateAccountView" in url or "CREATEACCOUNT" in combo:
                activity = "signup:finished"
            elif "/MyAccountView" in url:
                activity = "login_page"
            else:
                activity = ":signup:unknown"

        elif activity == "Home":
            temp = url.split("?")[0].replace("/us", "").lower()
            if temp == "/" or temp == "":
                # correct activity
                activity = "home"
            elif "error" in temp:
                activity = "error"
            elif "aboutus" in temp or "about-us" in temp:
                activity = "aboutus:static"
            elif "contactus" in temp or "contact-us" in temp:
                activity = "contactus:static"
            elif "customercare" in temp or "customer-care" in temp or "faq" in temp or "credit" in temp:
                activity = "customercare:static"
            elif "person" in temp or "registration" in temp:
                activity = "account"
            elif "logon" in temp or "logoff" in temp:
                activity = "account"
            elif "status" in temp:
                activity = "order_status"
            elif "style" in temp:
                activity = "sunglasses-by-style:clp"

        activity = activity.lower()
        if activity == "myaccount:static":
            if "UserOrderHistory" in url:
                activity = "order_status:history"

        # Actions are simpler -> we ignore most of them and the rest is already good
        if transform:
            # need to clean the activity
            cleaned_activity = self.transform_activity(activity, action)
            # if we have an action, append it to the activity
            if action is not None and pd.isna(action)==False:
                activity = activity + ":" + action
            return activity, cleaned_activity
        else:
            # no need to clean, just return
            if action is not None and pd.isna(action)==False:
                activity = activity + ":" + action
            return activity

    def transform_activity(self, activity: str, action: str = None, product_dict: dict = None) -> str:
        """Aggregates an activity into a larger group (including action if it is not None)"""
        if activity[-1] == ":":
            activity = activity[:-1]
        if activity[0] == ":":
            activity = activity[1:]

        # checkout process has 2 variants: standard and express
        if "checkout" in activity:
            splitted = activity.split(":")
            activity = splitted[0] + ":" + splitted[-1]

        # product pages
        if len(activity) > 4 and activity[-4:] == ":pdp":
            # replace sku with better name
            SKU = activity.split(":")[0]
            if product_dict and SKU in product_dict:
                activity = product_dict[SKU]
            else:
                activity = "product page"
        elif "brands:" in activity and ":plp" in activity:
            activity = "brand page"
        elif "cartpage" in activity:
            activity = "cart"
        elif len(activity) > 4 and activity[:5] == "remix" and activity[-4:] == ":pcp":
            activity = "customize sunglasses"
        elif ("sale" in activity or "clearance" in activity or "promo" in activity  # and need related word
              or "black-friday" in activity or "special-offers" in activity or 'cyber-monday' in activity
              or "discount" in activity):
            if "search" in activity and "_" not in activity:
                activity = "sale/discount search"
            else:
                activity = "sale/clearance/promo page"
        elif len(activity) > 5 and activity[-6:] == "search" or activity == "/searchdisplay":
            term = ":".join(activity.split(":")[:-1])
            term = re.sub(self.sunglasses_re, "", term)  # replace the word sunglass(es) in the term
            if "_" in term or "facet" in term:
                activity = "simple listing page"
            elif term in self.brand_names:
                activity = "brand search"
            elif self.product_match.match(term):
                # specific product (rb3025, etc.)
                if product_dict:
                    activity = term + " search"
                else:
                    activity = "suggested product search"
            elif is_integer(term):
                # search for SKU
                if product_dict and term in product_dict:
                    activity = product_dict[term] + " search"
                else:
                    activity = "suggested product search"
            elif "$" in term:
                # matches terms like ray-ban$212.00, which result from clicking a suggested product
                activity = "suggested product search"
            elif "_" not in term and ("sale" in term or "discount" in term):
                activity = "sale/discount search"
            else:
                activity = "custom search"
        elif ("account" in activity or "password" in activity or "login" in activity or "logon" in activity
              or "signup" in activity or "signon" in activity or "sign_in" in activity or "sign_on" in activity):
            activity = "account"
        elif "order_status" in activity:
            activity = "order status"
        elif "error" in activity or "404" in activity or "orphan" in activity:
            activity = "error"
        elif ("shipping-delivery" in activity or "terms" in activity or "policy" in activity
              or "returns" in activity or ("replacement" in activity and "oakley" not in activity)
              or "hto" in activity or "home-try-on" in activity):
            activity = "terms/conditions/policy info page"
        elif ("static" in activity or "customercare" in activity or "faq" in activity
              or "frequently" in activity or "guide" in activity or "site_map" in activity):
            activity = "static info page"
        elif "locator" in activity or "find-open-stores" in activity:
            activity = "find store page"
        elif (activity[-4:] == ":clp" or "face" in activity or "trends" in activity
              or "assetstore/exp" in activity or "?" in activity.split(":")[0]):
            activity = "editorial page"
        elif activity[-4:] == ":plp":
            activity = "simple listing page"
        elif activity[0] == "/" or activity[:3] == "us/":
            activity = "editorial page"

        # if we have an action (implies parse_actions = True, otherwise we would have action=None here)
        if action is not None and pd.isna(action)==False:
            if "slick-slide-control" in action:
                if activity != "editorial page" and activity != "cart":
                    activity += ":" + "scroll image"
                else:
                    return None
            else:
                activity += ":" + self.important_actions[action]
        elif self.parse_actions:
            # append pageview action if we parse other actions
            activity += ":pageview"

        return activity
