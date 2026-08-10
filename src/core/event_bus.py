"""
Async event bus for the trading system.
Central communication layer for all modules.
"""

import asyncio
from collections import defaultdict
import logging

from typing import Any, Awaitable, Callable, DefaultDict, List, TypeVar, Dict, Type


from pydantic_settings import BaseSettings
from core.events import (
    TickEvent, FeatureEvent, RegimeEvent, PatternEvent,
    PredictionEvent, ActionEvent, ExecutionEvent, FeedbackEvent,
    LearningEvent, EventType, EventHandler
)
from dataclasses import dataclass, field


@dataclass(slots=True)
class BusStats:
    published: int = 0
    processed: int = 0
    dropped: int = 0
    errors: int = 0


logger = logging.getLogger(__name__)

class EventBusConfig(BaseSettings):
    max_queue_size: int = 1024
    event_timeout_seconds: float = 2.0

class EventBus:
    """
    Central async event bus for the trading system.
    
    Handles subscription and publishing of events.
    All handlers are run as async tasks concurrently.
    """
    max_queue_size: int = 1024
    _subscribers: DefaultDict[EventType, list[EventHandler]] = field(default_factory=lambda: defaultdict(list))
    _queue: asyncio.Queue[Any] = field(init=False)
    _stats: BusStats = field(default_factory=BusStats)
    _running: bool = False
    
    def __init__(self, max_queue_size: int = 100):
        """
        Initialize event bus.
        
        Args:
            max_queue_size: Maximum number of pending events
        """
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.max_queue_size = max_queue_size
        self.event_queue = asyncio.Queue(maxsize=max_queue_size)
        self.active = False
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'errors': 0
        }
    
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: The event class to subscribe to
            handler: Async callable that handles the event
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type.__name__}")
    
    def unsubscribe(self, event_type: Type, handler: Callable) -> None:
        """
        Unsubscribe a handler from an event type.
        
        Args:
            event_type: The event class to unsubscribe from
            handler: The handler to remove
        """
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed {handler.__name__} from {event_type.__name__}")
    
    async def publish(self, event: Any) -> None:
        """
        Publish an event to all subscribers.
        
        Subscribers are called concurrently as async tasks.
        Does NOT wait for handlers to complete.
        
        Args:
            event: The event to publish
        """
        event_type = type(event)
        
        if event_type not in self.subscribers:
            logger.debug(f"No subscribers for {event_type.__name__}")
            return
        
        self.stats['events_published'] += 1
        handlers = self.subscribers[event_type]
        
        # Create concurrent tasks for all handlers
        tasks = []
        for handler in handlers:
            try:
                task = asyncio.create_task(handler(event))
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error creating task for {handler.__name__}: {e}")
                self.stats['errors'] += 1
        
        # Optionally wait for all handlers (fire-and-forget by default)
        # Uncomment to wait:
        # await asyncio.gather(*tasks, return_exceptions=True)
    
    async def publish_and_wait(self, event: Any) -> None:
        """
        Publish an event and wait for all handlers to complete.
        
        Args:
            event: The event to publish
        """
        event_type = type(event)
        
        if event_type not in self.subscribers:
            return
        
        self.stats['events_published'] += 1
        handlers = self.subscribers[event_type]
        
        tasks = []
        for handler in handlers:
            try:
                task = asyncio.create_task(handler(event))
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error creating task for {handler.__name__}: {e}")
                self.stats['errors'] += 1
        
        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.stats['events_processed'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'errors': 0
        }


# Global event bus instance
_bus: EventBus = None


def init_event_bus(max_queue_size: int = 100) -> EventBus:
    """Initialize the global event bus."""
    global _bus
    _bus = EventBus(max_queue_size=max_queue_size)
    return _bus


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _bus
    if _bus is None:
        _bus = init_event_bus()
    return _bus
