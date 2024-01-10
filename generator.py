import json
import datetime
import random
from typing import Any
from dateutil import relativedelta
from decimal import Decimal
from itertools import accumulate
import base58


def get_vesting_categories():
    return {
        # Coins for node runners, obtainable as rolls
        "node_running_coins": {
            "initial_release_ratio": Decimal("0.30"),
            "linear_vesting_duration": relativedelta.relativedelta(years=2),
            "obtainable_as_rolls": True
        },

        # Coins for ambassadors
        "ambassador_coins": {
            "initial_release_ratio": Decimal("0.30"),
            "linear_vesting_duration": relativedelta.relativedelta(years=2),
            "obtainable_as_rolls": False
        },

        # Coins for dashboard questers
        "quest_coins": {
            "initial_release_ratio": Decimal("1.0"),
            "linear_vesting_duration": relativedelta.relativedelta(years=0),
            "obtainable_as_rolls": False
        },

        # Coins for the main programs of the Massa Foundation (eg.community, airdrops, grants)
        "main_programs": {
            "initial_release_ratio": Decimal("0.05"),
            "linear_vesting_duration": relativedelta.relativedelta(years=5),
            "obtainable_as_rolls": False
        },

        # Coins for the public sale
        "public_sale": {
            "initial_release_ratio": Decimal("1.0"),
            "linear_vesting_duration": relativedelta.relativedelta(years=0),
            "obtainable_as_rolls": False
        },

        # Long-term locked coins for planned decentralization programs (eg. 100k nodes program)
        "decentralization_program": {
            "initial_release_ratio": Decimal("0.0"),
            "cliff_duration": relativedelta.relativedelta(years=2),
            "linear_vesting_duration": relativedelta.relativedelta(years=3),
            "obtainable_as_rolls": False
        },

        # Coins for private backers
        "backers": {
            "initial_release_ratio": Decimal("0.30"),
            "linear_vesting_duration": relativedelta.relativedelta(years=2),
            "obtainable_as_rolls": False
        },

        # Coins for private backers that did not provide their address in time.
        # Those coins are exclusively destined to be sent to the remaining backers
        # through a vesting smart contract following the same vesting scheme as "backers"
        # as soon as the remaining backer addresses are obtained.
        "late_backers_to_vest": {
            "initial_release_ratio": Decimal("1.0"),
            "linear_vesting_duration": relativedelta.relativedelta(years=0),
            "obtainable_as_rolls": False
        },

        # Remainder of the private distributions (eg. advisors, early grants)
        "remainder_private_distribution": {
            "initial_release_ratio": Decimal("0.30"),
            "linear_vesting_duration": relativedelta.relativedelta(years=2),
            "obtainable_as_rolls": False
        },

        # Coins for Massa Labs (eg. team, products, research, development)
        "massa_labs": {
            "initial_release_ratio": Decimal("0.0"),
            "linear_vesting_duration": relativedelta.relativedelta(years=5),
            "obtainable_as_rolls": False
        },

        # Coins for Massa Labs staking
        "massa_labs_staking": {
            "initial_release_ratio": Decimal("1.0"),
            "linear_vesting_duration": relativedelta.relativedelta(years=0),
            "obtainable_as_rolls": True
        },

        # Coins for the founders.
        # For legal reasons, the coins cannot be directly vested towards founder accounts.
        # Instead, this category will be fully moved to a triple vesting smart contract that
        # will make the coins progressively claimable by the founders.
        # The vesting scheme is: 5% initial then 5-year linear
        "founders": {
            "initial_release_ratio": Decimal("1.0"),
            "linear_vesting_duration": relativedelta.relativedelta(years=0),
            "obtainable_as_rolls": False
        }
    }


def datetime_to_massatime(dt):
    return int(datetime.datetime.timestamp(dt)*1000)


def massatime_to_datetime(mt):
    return datetime.datetime.fromtimestamp(mt / 1000, tz=datetime.timezone.utc)


def assert_address_is_valid(addr):
    if addr[0] != "A":
        raise Exception("Address does not start with A")
    if addr[1] not in ["U", "S"]:
        raise Exception("Address is not from a valid category")
    decoded = base58.b58decode_check(addr[2:])
    if decoded[0] != 0:
        raise Exception("Address version invalid")
    addr_hash = decoded[1:]
    if len(addr_hash) != 32:
        raise Exception("Address hash invalid")


# ensure determinism
random.seed(0)

# network parameters
thread_count = 32
t0_massatime = 16*1000
# launch date of the network
genesis_datetime = datetime.datetime(
    year=2024,
    month=1,
    day=15,
    hour=10,
    minute=0,
    second=0,
    microsecond=0,
    tzinfo=datetime.timezone.utc
)
genesis_timestamp = datetime_to_massatime(genesis_datetime)


class MassaAmount:
    def __init__(self, input):
        if isinstance(input, MassaAmount):
            self.dec = input.dec
        else:
            self.dec = Decimal(input)
        self.check()

    def divmod(self, other):
        if isinstance(other, MassaAmount):
            d, m = divmod(self.dec, other.dec)
        else:
            d, m = divmod(self.dec, Decimal(other))
        return int(d), MassaAmount(m)

    def __add__(self, other):
        if isinstance(other, MassaAmount):
            return MassaAmount(self.dec + other.dec)
        else:
            return MassaAmount(self.dec + Decimal(other))

    def __sub__(self, other):
        if isinstance(other, MassaAmount):
            return MassaAmount(self.dec - other.dec)
        else:
            return MassaAmount(self.dec - Decimal(other))

    def __mul__(self, other):
        return MassaAmount(self.dec * Decimal(other))

    # in place add
    def __iadd__(self, other):
        self.dec = (self + other).dec
        return self

    # in place subtract
    def __isub__(self, other):
        self.dec = (self - other).dec
        return self

    def mul_round(self, factor):
        return MassaAmount(round(self.dec * Decimal(factor), 9))

    def div_round(self, factor):
        return MassaAmount(round(self.dec / Decimal(factor), 9))

    # comparison operators

    def __eq__(self, other):
        return self.dec == other.dec

    def __ne__(self, other):
        return self.dec != other.dec

    def __lt__(self, other):
        return self.dec < other.dec

    def __le__(self, other):
        return self.dec <= other.dec

    def __gt__(self, other):
        return self.dec > other.dec

    def __ge__(self, other):
        return self.dec >= other.dec

    # check that the amount is valid

    def check(self):
        # check that the decimal is not negative
        if self.dec < 0:
            raise Exception("Negative decimal")

        # check that the decimal is not too big to be represented
        if self.dec > Decimal(18446744073.709551615):
            raise Exception("Decimal too big")

        # check precision loss
        quantized = self.dec.quantize(Decimal("1.000000000"))
        if quantized != self.dec:
            raise Exception("Precision loss detected.")
        self.dec = quantized

    def __str__(self):
        return format(self.dec, 'f')

    def __repr__(self):
        return str(self)

    def to_float(self):
        return float(self.dec)


class Slot:
    def __init__(self, period, thread):
        self.period = int(period)
        self.thread = int(thread)

    def __eq__(self, other):
        return self.period == other.period and self.thread == other.thread

    def __lt__(self, other):
        return self.period < other.period or (self.period == other.period and self.thread < other.thread)

    def __hash__(self):
        return hash((self.period, self.thread))

    # return the closest slot after a given massatime (included)
    def closest_slot_before_massatime(mt):
        if mt < genesis_timestamp:
            raise Exception("Datetime before launch")
        full_periods, remainder_mt = divmod(
            mt - genesis_timestamp, t0_massatime)
        full_threads = int(remainder_mt / (t0_massatime / thread_count))
        return Slot(full_periods, full_threads)

    # return the closest slot after a given datetime (included)
    def closest_slot_before_datetime(self, t):
        mt = datetime_to_massatime(t)
        return self.closest_slot_before_massatime(mt)

    def to_massatime(self):
        return genesis_timestamp + self.period * t0_massatime + self.thread * (t0_massatime / thread_count)

    def to_datetime(self):
        mt = self.to_massatime()
        return massatime_to_datetime(mt)

    def next(self):
        if self.thread == thread_count-1:
            return Slot(self.period+1, 0)
        else:
            return Slot(self.period, self.thread+1)

    def prev(self):
        if self.thread == 0:
            if self.period == 0:
                raise Exception("No previous slot")
            return Slot(self.period-1, thread_count-1)
        else:
            return Slot(self.period, self.thread-1)


def generate_vesting_events(coin_categories, disable_noise):
    # interval at which the linear vesting is released
    linear_release_interval = relativedelta.relativedelta(months=1)
    # do not schedule any events before a certain time after genesis
    minimal_wait = datetime.timedelta(days=2)

    # get the furthest vesting end date
    max_vesting_end_datetime = max([
        coin_category_info["vesting_end"]
        for coin_category_info in coin_categories.values()
    ])

    vesting_events = {}

    datetime_cursor = genesis_datetime
    prev_mt = datetime_to_massatime(genesis_datetime + minimal_wait)
    evt_index = 0
    while datetime_cursor <= max_vesting_end_datetime:
        evt_index += 1
        datetime_cursor = genesis_datetime + evt_index * linear_release_interval
        cursor_mt = datetime_to_massatime(datetime_cursor)
        if disable_noise is True:
            mt_noise = 0
        else:
            mt_noise = random.uniform(0, cursor_mt - prev_mt)
        event_slot = Slot.closest_slot_before_massatime(cursor_mt - mt_noise)
        vesting_events[event_slot] = {
            "datetime": event_slot.to_datetime(),
            "massatime": event_slot.to_massatime(),
            "amount": MassaAmount("0")
        }
        prev_mt = cursor_mt

    return vesting_events


# process one address
def process_addr(addr, addr_item, coin_categories):
    # check address validity
    assert_address_is_valid(addr)

    # create vesting events for this address
    vesting_events = generate_vesting_events(
        coin_categories, addr_item.get("disable_noise") or False)
    cliff_vesting_events = {}

    # initial coins not dedicated to staking for this address
    initial_nonstaking_coins = MassaAmount("0")

    # initial coins dedicated to staking for this address
    initial_staking_coins = MassaAmount("0")

    # iterate over all the coin categories for this address
    for coin_category_name, coin_category_amount_str in addr_item.items():
        # those are not really categories
        if coin_category_name in ["wants_initial_rolls", "disable_noise"]:
            continue

        # get the coin category information
        coin_category_info = coin_categories.get(coin_category_name)
        if coin_category_info is None:
            raise Exception(
                f"Unknown coin category {coin_category_name} for address {addr}")

        # get the amount of coins to credit this address for this category
        coin_category_amount = MassaAmount(coin_category_amount_str)

        # get the amount of coins released immediately for this coin category
        total_initial_release_for_category = coin_category_amount.mul_round(
            coin_category_info.get("initial_release_ratio") or Decimal("0"))

        # If the initial release of the coin category is obtainable as rolls, and the address wants that, then take it into account.
        # Otherwise, add it to the initial coins.
        if coin_category_info["obtainable_as_rolls"] is True and addr_item.get("wants_initial_rolls") is True:
            initial_staking_coins += total_initial_release_for_category
        else:
            initial_nonstaking_coins += total_initial_release_for_category

        # compute the amount of vested tokens remaining to schedule
        remaining_vested_coins = coin_category_amount - total_initial_release_for_category

        # select release event slots within the linear release range
        selected_event_slots = []

        if remaining_vested_coins > MassaAmount("0"):
            selected_event_slots = [
                s for s, v in vesting_events.items()
                if v["datetime"] > coin_category_info["cliff_end"] and v["datetime"] <= coin_category_info["vesting_end"]
            ]
        available_vesting_event_count = len(selected_event_slots)

        # handle the case where we don't do a linear release
        if remaining_vested_coins > MassaAmount("0") and available_vesting_event_count == 0:
            event_slot = Slot.closest_slot_before_datetime(
                coin_category_info["vesting_end"])
            if event_slot not in cliff_vesting_events:
                vesting_events[event_slot] = {
                    "datetime": event_slot.to_datetime(),
                    "massatime": event_slot.to_massatime(),
                    "amount": MassaAmount("0")
                }
            cliff_vesting_events[event_slot]["amount"] += remaining_vested_coins
            remaining_vested_coins = MassaAmount("0")

        # handle the case where we do a linear release
        if remaining_vested_coins > MassaAmount("0"):
            # split the vesting in chunks
            vesting_amount_per_event = remaining_vested_coins.div_round(
                available_vesting_event_count)
            for vesting_event_slot in selected_event_slots:
                evt = vesting_events[vesting_event_slot]
                if remaining_vested_coins >= vesting_amount_per_event:
                    evt["amount"] += vesting_amount_per_event
                    remaining_vested_coins -= vesting_amount_per_event
                else:
                    evt["amount"] += remaining_vested_coins
                    remaining_vested_coins = MassaAmount("0")
                    break

            # if there is anything remaining, put it in the last vesting event
            if remaining_vested_coins > MassaAmount("0"):
                evt = vesting_events[max(selected_event_slots)]
                evt["amount"] += remaining_vested_coins
                remaining_vested_coins = MassaAmount("0")

    # convert the staking coins to rolls, send the remainder to initial nonstaking coins
    roll_price = MassaAmount("100")
    initial_rolls, remainder = initial_staking_coins.divmod(roll_price)
    initial_nonstaking_coins = initial_nonstaking_coins + remainder

    # fuse cliff_vesting_events into vesting events
    for cliff_vesting_event_slot, cliff_vesting_event in cliff_vesting_events.items():
        if cliff_vesting_event_slot not in vesting_events:
            vesting_events[cliff_vesting_event_slot] = cliff_vesting_event
        else:
            vesting_events[cliff_vesting_event_slot]["amount"] += cliff_vesting_event["amount"]

    # add the vesting events to the deferred credits
    deferred_credits = {}
    for vesting_event_slot, vesting_event_value in vesting_events.items():
        amount = vesting_event_value["amount"]
        if amount > MassaAmount("0"):
            if vesting_event_slot not in deferred_credits:
                deferred_credits[vesting_event_slot] = MassaAmount("0")
            deferred_credits[vesting_event_slot] += amount

    # convert deferred credits to a sorted, formatted list
    deferred_credits = list(deferred_credits.items())
    deferred_credits.sort(key=lambda x: x[0])
    deferred_credits = [
        {
            "slot": {
                "period": int(slot.period),
                "thread": int(slot.thread)
            },
            "amount": str(amount)
        }
        for slot, amount in deferred_credits
    ]
    if len(deferred_credits) == 0:
        deferred_credits = None

    # format initial ledger entry
    initial_ledger_entry = None
    if initial_nonstaking_coins > MassaAmount("0"):
        initial_ledger_entry = {
            "balance": str(initial_nonstaking_coins),
            "datastore": [],
            "bytecode": []
        }

    # format initial_rolls
    if initial_rolls == 0:
        initial_rolls = None

    return initial_ledger_entry, initial_rolls, deferred_credits


# generate the initial node files
def generate_initial_node_files(input_paths):
    with open("node_initial_setup/genesis_timestamp.json", "w") as f:
        json.dump({
            "genesis_timestamp": genesis_timestamp
        }, f, indent=1)
    print("Genesis timestamp:", genesis_timestamp)

    # list of coin categories, each with its vesting parameters
    coin_categories = get_vesting_categories()

    # compute cliff and vesting end times
    for coin_category_info in coin_categories.values():
        coin_category_info["cliff_end"] = genesis_datetime + (coin_category_info.get(
            "cliff_duration") or relativedelta.relativedelta(seconds=0))
        coin_category_info["vesting_end"] = coin_category_info["cliff_end"] + (
            coin_category_info.get("linear_vesting_duration") or relativedelta.relativedelta(seconds=0))

    # aggregate all the data from the input paths
    input_data = {}
    for input_path in input_paths:
        file_data = {}
        with open(input_path, "r") as f:
            file_data = json.load(f)
        for addr in file_data:
            if addr not in input_data:
                input_data[addr] = {}
            input_data[addr] = {
                **input_data[addr],
                **file_data[addr]
            }
    # for determinism, turn the dict into a list of tuples sorted by address
    input_data = list(input_data.items())
    input_data.sort(key=lambda x: x[0])

    # initial ledger
    initial_ledger = {}

    # deferred credits (vesting schedule)
    initial_deferred_credits = {}

    # initial rolls
    initial_rolls = {}

    # iterate over all addresses
    n_addrs = len(input_data)
    for addr_i, (addr, addr_item) in enumerate(input_data):
        if addr_i % 100 == 0:
            print(f"Processing address {addr_i} ({n_addrs-addr_i} remaining)")
        addr_initial_ledger_entry, addr_initial_rolls, addr_deferred_credits = process_addr(
            addr, addr_item, coin_categories
        )
        if addr_initial_ledger_entry is not None:
            initial_ledger[addr] = addr_initial_ledger_entry
        if addr_deferred_credits is not None:
            initial_deferred_credits[addr] = addr_deferred_credits
        if addr_initial_rolls is not None:
            initial_rolls[addr] = addr_initial_rolls

    # save initial ledger
    with open("node_initial_setup/initial_ledger.json", "w") as f:
        json.dump(initial_ledger, f, sort_keys=True, indent=1)

    # save initial rolls
    with open("node_initial_setup/initial_rolls.json", "w") as f:
        json.dump(initial_rolls, f, sort_keys=True, indent=1)

    # save initial deferred credits
    with open("node_initial_setup/deferred_credits.json", "w") as f:
        json.dump(initial_deferred_credits, f, sort_keys=True, indent=1)


# generate
generate_initial_node_files([
    "input_listings/dashboard_data.json",
    "input_listings/foundation.json",
    "input_listings/backers.json",
    "input_listings/labs.json",
    "input_listings/founders.json"
])


# check total supply
def check_total_supply():
    with open("node_initial_setup/initial_ledger.json", "r") as f:
        initial_ledger = sum([MassaAmount(v["balance"])
                             for _, v in json.load(f).items()], MassaAmount("0"))
    with open("node_initial_setup/initial_rolls.json", "r") as f:
        initial_rolls = sum([MassaAmount("100") * Decimal(v)
                            for _, v in json.load(f).items()], MassaAmount("0"))
    with open("node_initial_setup/deferred_credits.json", "r") as f:
        deferred_credits = sum([sum([MassaAmount(v["amount"]) for v in u], MassaAmount(
            "0")) for _, u in json.load(f).items()], MassaAmount("0"))
    total_supply = initial_ledger + initial_rolls + deferred_credits
    expected_total_supply = MassaAmount("1000000000")
    print("TOTAL SUPPLY:", total_supply)
    if total_supply != expected_total_supply:
        raise Exception(
            f"Total supply is {total_supply} but should be {expected_total_supply}")


check_total_supply()
