from carbonserver.api import schemas

import abc


class Team(abc.ABC):
    @abc.abstractmethod
    def save_organization(self, organization: schemas.OrganizationCreate):
        raise NotImplementedError

    @abc.abstractmethod
    def get_one_organization(self, organization_id):
        raise NotImplementedError

    @abc.abstractmethod
    def get_team_from_organizations(self, organization_id):
        raise NotImplementedError
