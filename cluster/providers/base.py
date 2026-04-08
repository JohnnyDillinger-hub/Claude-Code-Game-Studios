from __future__ import annotations

from abc import ABC, abstractmethod

from cluster.providers.models import BootstrapBundle, ProviderBlueprint, ProviderOffer, ProvisionRequest, ProvisionedResource


class ProviderError(RuntimeError):
    """Raised when provider discovery or provisioning fails."""


class ProviderAdapter(ABC):
    name: str

    @abstractmethod
    def list_offers(self) -> list[ProviderOffer]:
        raise NotImplementedError

    @abstractmethod
    def validate_request(
        self,
        request: ProvisionRequest,
        blueprint: ProviderBlueprint | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_resource(
        self,
        request: ProvisionRequest,
        bundle: BootstrapBundle,
        blueprint: ProviderBlueprint | None = None,
        selected_offer: ProviderOffer | None = None,
    ) -> ProvisionedResource:
        raise NotImplementedError
