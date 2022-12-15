import datetime


class ResultManager:

    FILENAME = "diary.txt"

    @staticmethod
    def save_result(algorithm, distancestsp_size, individual):
        with open(ResultManager.FILENAME, "a+") as file:
            file.write(
                f"{datetime.datetime.now()} - {algorithm} - {algorithm} - TSP{distancestsp_size} - {individual}\n"
            )
