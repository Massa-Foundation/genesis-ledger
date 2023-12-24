import json
import datetime
from decimal import *
import random
from dateutil import relativedelta

# ensure determinism
random.seed(0)

# check that a decimal representing a coin amount is valid
def check_amount(d):
    # check that the decimal is not negative
    if d < 0:
        raise Exception("Negative decimal")
    
    # check that the decimal is not too big to be represented
    if d > Decimal(18446744073.709551615):
        raise Exception("Decimal too big")
    
    # check precision loss
    if d.quantize(Decimal("1.000000000")) != d:
        raise Exception("Precision loss detected.")
    

def datetime_to_massatime(dt):
    return int(datetime.datetime.timestamp(dt)*1000)

def massatime_to_slot(launch_massatime, mt):
    # thread count
    thread_count = 32

    # t0
    t0_massatime = 16*1000

    period, remainder = divmod(mt - launch_massatime, t0_massatime)
    thread = int(remainder / (t0_massatime / thread_count))

    return {
        "period": period,
        "thread": thread
    }


# generate the initial node files
def generate_initial_node_files(input_paths):
    # launch date of the network
    launch_date = datetime.datetime(
        year=2024,
        month=1,
        day=15,
        hour=10,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=datetime.timezone.utc
    )
    launch_massatime = datetime_to_massatime(launch_date)

    with open("node_initial_setup/launch_timestamp.json", "w") as f:
        json.dump({
            "launch_timestamp": launch_massatime
        }, f, indent=1)
    print("Launch massatime:", launch_massatime)

    # list of coin categories, each with its vesting parameters
    coin_categories = {
        "node_running_coins": {
            "initial_release_ratio": Decimal("0.30"),
            "linear_vesting_duration": (launch_date + relativedelta.relativedelta(years=2)) - launch_date,  # we do this because not all years have the same number of days
            "obtainable_as_rolls": True
        },
        "ambassador_coins": {
            "initial_release_ratio": Decimal("0.30"),
            "linear_vesting_duration": (launch_date + relativedelta.relativedelta(years=2)) - launch_date,
            "obtainable_as_rolls": False
        },
        "quest_coins": {
            "initial_release_ratio": Decimal("0.30"),
            "linear_vesting_duration": (launch_date + relativedelta.relativedelta(years=2)) - launch_date,
            "obtainable_as_rolls": False
        }
    }
    max_vesting_duration = max([coin_category_info["linear_vesting_duration"] for coin_category_info in coin_categories.values()])

    # interval at which the linear vesting is released
    linear_release_interval = datetime.timedelta(days=7)

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

    # initial coin
    initial_coins = {}

    # deferred credits (vesting schedule)
    initial_deferred_credits = {}

    # initial rolls
    initial_rolls = {}

    # iterate over all addresses
    for addr, addr_item in input_data.items():
        # initial coins dedicated to staking to this address
        initial_staking_coins = Decimal("0")

        # pre-compute the vesting event dates for this address
        # add some random wiggle to the vesting event dates to avoid having all the vesting events on the same day
        vesting_event_count, remaining_vesting_time = divmod(max_vesting_duration, linear_release_interval)
        if remaining_vesting_time > datetime.timedelta(0):
            vesting_event_count += 1
        vesting_event_dates = []
        vesting_event_values = []
        minimal_wait = datetime.timedelta(days=2)  # do not schedule any events before 2 days after genesis
        for i in range(vesting_event_count):
            noise = datetime.timedelta(seconds=random.uniform(0, linear_release_interval.total_seconds()))
            event_date = launch_date + linear_release_interval * (i+1) - noise
            while event_date < launch_date + minimal_wait:
                noise = datetime.timedelta(seconds=random.uniform(0, linear_release_interval.total_seconds()))
                event_date = launch_date + linear_release_interval * (i+1) - noise
            vesting_event_dates.append(event_date)
            vesting_event_values.append(Decimal("0"))

        # iterate over all the coin categories for this address
        for coin_category_name, coin_category_amount_str in addr_item.items():
            if coin_category_name == "wants_initial_rolls":
                continue

            # get the coin category information
            coin_category_info = coin_categories.get(coin_category_name)
            if coin_category_info is None:
                raise Exception(f"Unknown coin category {coin_category_name} for address {addr}")

            # get the amount of coins to credit this address for this category
            coin_category_amount = Decimal(coin_category_amount_str)
            check_amount(coin_category_amount)
            
            # create element in initial ledger if it is not there
            if addr not in initial_coins:
                initial_coins[addr] = Decimal("0")
            
            # get the amount of coins released immediately for this coin category
            total_initial_release_for_category = round(Decimal(coin_category_info.get("initial_release_ratio") or Decimal("0")) * coin_category_amount, 9)
            check_amount(total_initial_release_for_category)

            # If the initial release of the coin category is obtainable as rolls, and the address that, then take it into account.
            # Otherwise, add it to the initial coins.
            if coin_category_info["obtainable_as_rolls"] is True and addr_item.get("wants_initial_rolls") is True:
                initial_staking_coins += total_initial_release_for_category
                check_amount(initial_staking_coins)
            else:
                initial_coins[addr] += total_initial_release_for_category
                check_amount(initial_coins[addr])

            # compute the amount of vested tokens remaining to schedule
            remaining_vested_coins = coin_category_amount - total_initial_release_for_category
            check_amount(remaining_vested_coins)

            # spread the vested tokens into the vesting events
            if remaining_vested_coins > Decimal("0"):
                # count all available vesting events to distribute to
                available_vesting_event_count = len([
                    v for v in vesting_event_dates if v <= launch_date + coin_category_info["linear_vesting_duration"]
                ])
                if available_vesting_event_count == 0:
                    raise Exception(f"Invalid vesting schedule for address {addr}")

                # split the vesting in chunks
                vesting_amount_per_event = round(remaining_vested_coins / available_vesting_event_count, 9)
                check_amount(vesting_amount_per_event)
                for vesting_event_index in range(available_vesting_event_count):
                    if remaining_vested_coins >= vesting_amount_per_event:
                        vesting_event_values[vesting_event_index] += vesting_amount_per_event
                        check_amount(vesting_event_values[vesting_event_index])
                        remaining_vested_coins -= vesting_amount_per_event
                    else:
                        vesting_event_values[vesting_event_index] += remaining_vested_coins
                        check_amount(vesting_event_values[vesting_event_index])
                        remaining_vested_coins = Decimal("0")
                
                # if there is anything remaining, put it in the last vesting event
                if remaining_vested_coins > Decimal("0"):
                    vesting_event_values[available_vesting_event_count-1] = remaining_vested_coins
                    check_amount(vesting_event_values[available_vesting_event_count-1])
                    remaining_vested_coins = Decimal("0")
        
        # convert the staking coins to rolls, send the remainder to initial coins
        roll_price = Decimal("100")
        roll_count, remainder = divmod(initial_staking_coins, roll_price)
        roll_count = int(roll_count)
        if roll_count > 0:
            initial_rolls[addr] = roll_count
        if remainder > Decimal("0"):
            if addr not in initial_coins:
                initial_coins[addr] = remainder
            else:
                initial_coins[addr] += remainder
            check_amount(initial_coins[addr])
        
        # add the vesting events to the deferred credits
        for vesting_event_date, vesting_event_value in zip(vesting_event_dates, vesting_event_values):
            if vesting_event_value > Decimal("0"):
                if addr not in initial_deferred_credits:
                    initial_deferred_credits[addr] = []
                check_amount(vesting_event_value)
                initial_deferred_credits[addr].append({
                    "date": vesting_event_date,
                    "amount": vesting_event_value
                })
        
    # save initial coins
    formatted_initial_coins = {
        addr: {
            "balance": str(balance),
            "datastore": [],
            "bytecode": []
        } for addr, balance in initial_coins.items()
    }
    with open("node_initial_setup/initial_ledger.json", "w") as f:
        json.dump(formatted_initial_coins, f, indent=1)

    # save initial rolls
    with open("node_initial_setup/initial_rolls.json", "w") as f:
        json.dump(initial_rolls, f, indent=1)

    # format initial deferred credits
    formatted_initial_deferred_credits = {}
    for addr, addr_deferred_credits in initial_deferred_credits.items():
        formatted_initial_deferred_credits[addr] = []
        for deferred_credit in addr_deferred_credits:
            formatted_initial_deferred_credits[addr].append({
                "slot": massatime_to_slot(launch_massatime, datetime_to_massatime(deferred_credit["date"])),
                "amount": str(deferred_credit["amount"])
            })
    with open("node_initial_setup/deferred_credits.json", "w") as f:
        json.dump(formatted_initial_deferred_credits, f, indent=1)

generate_initial_node_files([
    "input_listings/dashboard_data.json"
])


# TODO perform checks
