from datetime import datetime

def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[RUN] Script executed at: {now}")

    # simulate output file
    with open("log.txt", "a") as f:
        f.write(f"Executed at: {now}\n")

main()