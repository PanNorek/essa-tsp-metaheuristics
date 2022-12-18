import datetime


class ResultManager:

    FILENAME = "diary.txt"

    @staticmethod
    def save_result(algorithm, distancestsp_size, individual):
        with open(ResultManager.FILENAME, "a+") as file:
            file.write(
                f"DATE: {datetime.datetime.now()}\nSPECIFICATION: {algorithm}\nSIZE: TSP{distancestsp_size}\nDISTANCE: {individual.distance}\nPATH: {individual.path}\n"
            )
