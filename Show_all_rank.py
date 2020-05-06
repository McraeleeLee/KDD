import subprocess
import json
class Team:
    def __init__(self, data):
        org = data.get('teamLeaderOrganization', 'xx')
        schools = '&'.join(sorted(set([m['graduateSchool'] for m in data['teamMemberList'] if 'graduateSchool' in m])))
        self.team_name = data['teamName'] + (f' @{org}' if len(org) else '') + (f' [{schools}]' if schools else '')
        self.score_rank = data['rank']
        self.score_object = data['scoreJsonObject']
        self.hitrate_50_full = data['scoreJsonObject']['hitrate_50_full']
        self.hitrate_50_half = data['scoreJsonObject']['hitrate_50_half']
        self.ndcg_50_full = data['scoreJsonObject']['ndcg_50_full']
        self.ndcg_50_half = data['scoreJsonObject']['ndcg_50_half']
        self.submit_time = data['gmtSubmit']

    @staticmethod
    def line_format():
        return '{:^10} {:^10} {:^18} {:^18} {:^18} {:^18} {:^20} {}'

    @staticmethod
    def line_header():
        header = [
            'new_rank',
            'score_rank',
            'hitrate_full',
            'ndcg_full',
            'hitrate_half',
            'ndcg_half',
            'submit_time',
            'team_name',
        ]
        return Team.line_format().format(*header)

    def to_line(self, new_rank):
        line = [
            str(new_rank),
            str(self.score_rank),
            str(self.hitrate_50_full)[:16],
            str(self.ndcg_50_full)[:16],
            str(self.hitrate_50_half)[:16],
            str(self.ndcg_50_half)[:16],
            self.submit_time,
            self.team_name,
        ]
        return Team.line_format().format(*line)

def show_Leaderboard(ranking_by_key='ndcg_50_half'):
    url = 'https://tianchi.aliyun.com/mobile/api/proxy/competitionService/api/race/queryRankingList?season=0&raceId=231785&pageIndex=1&max=5000'
    with subprocess.Popen(f'curl "{url}"', shell=True, stdout=subprocess.PIPE) as p:
        for line in p.stdout:
            line = line.decode('utf-8')
            data = json.loads(line)
            ranking_list = data['data']['list']
    team_list = [Team(t) for t in ranking_list]
    team_list = sorted(team_list, key=lambda t: -t.score_object[ranking_by_key])
    print(Team.line_header())
    for i, t in enumerate(team_list):
        print(t.to_line(i + 1))


def main():
    show_Leaderboard()


if __name__ == '__main__':
    main()