import asyncio
import sys
import bleakheart as bh
from bleak import BleakClient

ADDRESS = "24:AC:AC:13:C4:6E"

async def run():
    print(f"Testing bare-metal connection to {ADDRESS}...")
    try:
        async with BleakClient(ADDRESS, timeout=15.0) as client:
            print("[+] Connected to BleakClient.")
            
            print("[.] Forcing explicit pair() to fix Insufficient Auth...")
            try:
                res = await client.pair()
                print(f"[+] Pair result: {res}")
            except Exception as e:
                print(f"[-] Pair failed: {type(e).__name__} -> {e}")
                
            ecg_queue = asyncio.Queue()
            hr_queue = asyncio.Queue()
            pmd = bh.PolarMeasurementData(client, ecg_queue=ecg_queue)
            heartrate = bh.HeartRate(client, queue=hr_queue)
            
            try:
                meas = await asyncio.wait_for(pmd.available_measurements(), timeout=10)
                print(f"[+] PMD Capabilities: {meas}")
            except Exception as e:
                print(f"[!] PMD Query Failed: {e}")
                
            print("[.] Attempting heartrate.start_notify()...")
            task = asyncio.create_task(heartrate.start_notify())
            try:
                await asyncio.wait_for(task, timeout=10.0)
                print("[+] heartrate.start_notify() succeeded!")
            except asyncio.TimeoutError:
                print("[!] TIMEOUT: heartrate.start_notify() hung for 10 seconds!")
                return
                
    except Exception as e:
        print(f"\n[!] CRASH: {type(e).__name__} -> {e}")

if __name__ == "__main__":
    asyncio.run(run())
