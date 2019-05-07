import generate_csv
import train


def main():
    # train.train()
    gcsv = generate_csv.Generate_CSV()
    gcsv.process_all()


if __name__ == '__main__':
    main()
