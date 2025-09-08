from preprocessing.investing.make_long_csv import MakeLongCSV

file_path = '../../investing_data_file/경제활동별_GDP_및_GNI_계절조정__실질__분기__20250908011659.csv'
processer = MakeLongCSV(file_path)

processed = (
    processer
    .filter_data("계정항목별", ["정보통신업"], filename="filtered_info_comm.csv")
    .to_long(
        id_vars=["계정항목별"], filename="info_comm_long.csv"
    )
    .get_data()
)
