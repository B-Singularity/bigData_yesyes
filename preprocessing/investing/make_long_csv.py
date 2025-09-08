import pandas as pd
import os

class MakeLongCSV:

    def __init__(self, file_path: str, output_dir: str = "outputs"):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        print("[초기 데이터 미리보기]")
        print(self.df.head(), "\n")
        self.save_output("initial.csv")

    def filter_data(self, column: str, values: list, filename: str = "filtered.csv"):
        """
        특정 컬럼에서 지정된 값만 남기도록 필터링
        """
        self.df = self.df[self.df[column].isin(values)]
        print(f"[{column} 필터링 결과: {values}]")
        print(self.df.head(), "\n")
        self.save_output(filename)
        return self

    def to_long(self, id_vars: list,
                var_name: str = "기간", value_name: str = "값",
                filename: str = "long.csv"):
        """
        wide → long 변환 (id_vars 제외한 모든 컬럼을 기간으로 인식)
        """
        value_vars = [c for c in self.df.columns if c not in id_vars]

        self.df = pd.melt(
            self.df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )
        print("[long format 변환 결과]")
        print(self.df.head(), "\n")
        self.save_output(filename)
        return self

    def save_output(self, filename: str):
        """
        현재 DataFrame을 CSV로 저장
        """
        path = os.path.join(self.output_dir, filename)
        self.df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[저장 완료] {path}")

    def get_data(self):
        """
        처리된 DataFrame 반환
        """
        return self.df.copy()

