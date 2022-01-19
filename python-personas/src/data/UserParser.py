from typing import List, Tuple


def parse_location(location_str: str) -> list:
    """Parses a location ('<city (<area>, <country>)') into the three different parts"""
    return location_str.replace(" (", ", ")[:-1].split(", ")


def get_header():
    """Returns the header of the user table"""
    return ["user_id", "visitor_id", "city", "area", "country", "device", "os", "ltt"]


class UserParser:

    def __init__(self):
        self.id = -1

    def parse_user(self, row) -> Tuple[int, List]:
        """"Parses a row into a user, keeping track of an id mapping. It returns an id and the row itself"""
        visitor_id = row["visitor_id"]
        location = parse_location(row["location"])
        device = row["device"]
        os = row["os"]
        ltt = row["ltt"]

        self.id += 1
        user = [self.id, visitor_id, location[0], location[1], location[2], device, os, ltt]
        return self.id, user
