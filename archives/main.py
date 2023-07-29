import aiuci


def main() -> None:
    uciLoop = aiuci.UCI()

    while True:
        command = input()
        uciLoop.processCommand(command)

        if command == "quit":
            break


if __name__ == "__main__":
    main()
