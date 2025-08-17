#!/usr/bin/env python3
"""
Real-time Data Streaming Module

This module provides real-time financial data streaming capabilities for the GMF
Time Series Forecasting system, enabling live data ingestion and processing.

Author: GMF Investment Team
Version: 2.0.0
"""

import asyncio
import aiohttp
import websockets
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from queue import Queue, Empty
import warnings

logger = logging.getLogger(__name__)


class RealTimeDataStreamer:
    """
    Real-time financial data streaming with multiple data source support.

    Features:
    - WebSocket connections for live data
    - REST API polling for real-time updates
    - Data validation and quality checks
    - Automatic reconnection and error handling
    - Configurable data processing pipelines
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real-time data streamer.

        Args:
            config: Configuration dictionary for streaming settings
        """
        self.config = config or self._get_default_config()
        self.connections = {}
        self.data_queues = {}
        self.processors = {}
        self.is_running = False
        self.callbacks = []
        self.error_handlers = []

        # Initialize data storage
        self.latest_data = {}
        self.data_history = {}
        self.connection_status = {}

        logger.info("Real-time data streamer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for streaming."""
        return {
            'websocket_timeout': 30,
            'reconnect_attempts': 5,
            'reconnect_delay': 5,
            'max_queue_size': 10000,
            'data_retention_hours': 24,
            'batch_size': 100,
            'enable_compression': True,
            'log_level': 'INFO'
        }

    def add_data_source(self, source_name: str, source_config: Dict[str, Any]) -> bool:
        """
        Add a new data source for streaming.

        Args:
            source_name: Unique name for the data source
            source_config: Configuration for the data source

        Returns:
            bool: True if source added successfully
        """
        try:
            # Validate source configuration
            required_fields = ['type', 'url', 'symbols']
            if not all(field in source_config for field in required_fields):
                logger.error(
                    f"Missing required fields for source '{source_name}'")
                return False

            # Initialize data structures
            self.data_queues[source_name] = Queue(
                maxsize=self.config['max_queue_size'])
            self.latest_data[source_name] = {}
            self.data_history[source_name] = []
            self.connection_status[source_name] = 'disconnected'

            # Store source configuration
            self.connections[source_name] = source_config

            logger.info(
                f"Added data source '{source_name}' with {len(source_config['symbols'])} symbols")
            return True

        except Exception as e:
            logger.error(
                f"Failed to add data source '{source_name}': {str(e)}")
            return False

    def add_data_processor(self, processor_name: str, processor_func: Callable) -> bool:
        """
        Add a custom data processor function.

        Args:
            processor_name: Name for the processor
            processor_func: Function to process incoming data

        Returns:
            bool: True if processor added successfully
        """
        try:
            self.processors[processor_name] = processor_func
            logger.info(f"Added data processor '{processor_name}'")
            return True
        except Exception as e:
            logger.error(
                f"Failed to add data processor '{processor_name}': {str(e)}")
            return False

    def add_callback(self, callback_func: Callable) -> bool:
        """
        Add a callback function for data updates.

        Args:
            callback_func: Function to call when new data arrives

        Returns:
            bool: True if callback added successfully
        """
        try:
            self.callbacks.append(callback_func)
            logger.info("Added data update callback")
            return True
        except Exception as e:
            logger.error(f"Failed to add callback: {str(e)}")
            return False

    def add_error_handler(self, error_handler: Callable) -> bool:
        """
        Add an error handler function.

        Args:
            error_handler: Function to handle errors

        Returns:
            bool: True if error handler added successfully
        """
        try:
            self.error_handlers.append(error_handler)
            logger.info("Added error handler")
            return True
        except Exception as e:
            logger.error(f"Failed to add error handler: {str(e)}")
            return False

    async def start_streaming(self) -> bool:
        """
        Start real-time data streaming for all sources.

        Returns:
            bool: True if streaming started successfully
        """
        if self.is_running:
            logger.warning("Streaming already running")
            return True

        try:
            self.is_running = True

            # Start streaming tasks for each source
            streaming_tasks = []
            for source_name, source_config in self.connections.items():
                if source_config['type'] == 'websocket':
                    task = asyncio.create_task(
                        self._websocket_stream(source_name, source_config))
                elif source_config['type'] == 'rest':
                    task = asyncio.create_task(
                        self._rest_polling_stream(source_name, source_config))
                else:
                    logger.warning(
                        f"Unknown source type '{source_config['type']}' for '{source_name}'")
                    continue

                streaming_tasks.append(task)

            # Start data processing task
            processing_task = asyncio.create_task(self._process_data_loop())

            # Start cleanup task
            cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info(
                f"Started streaming for {len(streaming_tasks)} data sources")
            return True

        except Exception as e:
            logger.error(f"Failed to start streaming: {str(e)}")
            self.is_running = False
            return False

    async def stop_streaming(self) -> bool:
        """
        Stop real-time data streaming.

        Returns:
            bool: True if streaming stopped successfully
        """
        try:
            self.is_running = False

            # Close all connections
            for source_name in self.connections:
                self.connection_status[source_name] = 'disconnected'

            logger.info("Stopped real-time data streaming")
            return True

        except Exception as e:
            logger.error(f"Failed to stop streaming: {str(e)}")
            return False

    async def _websocket_stream(self, source_name: str, source_config: Dict[str, Any]):
        """Handle WebSocket streaming for a data source."""
        reconnect_attempts = 0
        max_attempts = self.config['reconnect_attempts']

        while self.is_running and reconnect_attempts < max_attempts:
            try:
                self.connection_status[source_name] = 'connecting'

                # Connect to WebSocket
                async with websockets.connect(
                    source_config['url'],
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:

                    self.connection_status[source_name] = 'connected'
                    reconnect_attempts = 0
                    logger.info(
                        f"WebSocket connected for source '{source_name}'")

                    # Subscribe to symbols if needed
                    if 'subscribe_message' in source_config:
                        await websocket.send(json.dumps(source_config['subscribe_message']))

                    # Stream data
                    async for message in websocket:
                        if not self.is_running:
                            break

                        try:
                            # Parse message
                            data = json.loads(message)

                            # Add to queue
                            if not self.data_queues[source_name].full():
                                self.data_queues[source_name].put({
                                    'source': source_name,
                                    'timestamp': datetime.now().isoformat(),
                                    'data': data
                                })
                            else:
                                logger.warning(
                                    f"Queue full for source '{source_name}', dropping message")

                        except json.JSONDecodeError:
                            logger.warning(
                                f"Invalid JSON from source '{source_name}'")
                        except Exception as e:
                            logger.error(
                                f"Error processing message from '{source_name}': {str(e)}")

            except websockets.exceptions.ConnectionClosed:
                logger.info(
                    f"WebSocket connection closed for source '{source_name}'")
            except Exception as e:
                logger.error(
                    f"WebSocket error for source '{source_name}': {str(e)}")

            # Handle reconnection
            if self.is_running:
                reconnect_attempts += 1
                self.connection_status[source_name] = 'reconnecting'
                logger.info(
                    f"Attempting to reconnect to '{source_name}' (attempt {reconnect_attempts})")
                await asyncio.sleep(self.config['reconnect_delay'])

    async def _rest_polling_stream(self, source_name: str, source_config: Dict[str, Any]):
        """Handle REST API polling for a data source."""
        poll_interval = source_config.get('poll_interval', 1)

        while self.is_running:
            try:
                self.connection_status[source_name] = 'polling'

                # Fetch data from REST API
                async with aiohttp.ClientSession() as session:
                    for symbol in source_config['symbols']:
                        if not self.is_running:
                            break

                        try:
                            # Construct URL
                            url = source_config['url'].format(symbol=symbol)

                            # Make request
                            async with session.get(url, timeout=10) as response:
                                if response.status == 200:
                                    data = await response.json()

                                    # Add to queue
                                    if not self.data_queues[source_name].full():
                                        self.data_queues[source_name].put({
                                            'source': source_name,
                                            'symbol': symbol,
                                            'timestamp': datetime.now().isoformat(),
                                            'data': data
                                        })
                                    else:
                                        logger.warning(
                                            f"Queue full for source '{source_name}', dropping data")
                                else:
                                    logger.warning(
                                        f"HTTP {response.status} from source '{source_name}' for symbol '{symbol}'")

                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Timeout fetching data for symbol '{symbol}' from '{source_name}'")
                        except Exception as e:
                            logger.error(
                                f"Error fetching data for symbol '{symbol}' from '{source_name}': {str(e)}")

                # Wait before next poll
                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error(
                    f"REST polling error for source '{source_name}': {str(e)}")
                await asyncio.sleep(poll_interval)

    async def _process_data_loop(self):
        """Main data processing loop."""
        while self.is_running:
            try:
                # Process data from all sources
                for source_name in self.connections:
                    try:
                        # Get data from queue (non-blocking)
                        while not self.data_queues[source_name].empty():
                            data_item = self.data_queues[source_name].get_nowait(
                            )

                            # Process data
                            processed_data = await self._process_data_item(data_item)

                            if processed_data:
                                # Update latest data
                                self.latest_data[source_name] = processed_data

                                # Add to history
                                self.data_history[source_name].append(
                                    processed_data)

                                # Call callbacks
                                await self._notify_callbacks(processed_data)

                    except Empty:
                        pass  # Queue is empty
                    except Exception as e:
                        logger.error(
                            f"Error processing data from '{source_name}': {str(e)}")
                        await self._handle_error(e, source_name)

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in data processing loop: {str(e)}")
                await asyncio.sleep(1)

    async def _process_data_item(self, data_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single data item."""
        try:
            processed_data = data_item.copy()

            # Apply custom processors
            for processor_name, processor_func in self.processors.items():
                try:
                    processed_data = await processor_func(processed_data)
                except Exception as e:
                    logger.error(
                        f"Error in processor '{processor_name}': {str(e)}")

            # Validate processed data
            if self._validate_data(processed_data):
                return processed_data
            else:
                logger.warning(
                    f"Data validation failed for source '{data_item.get('source', 'unknown')}'")
                return None

        except Exception as e:
            logger.error(f"Error processing data item: {str(e)}")
            return None

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate processed data."""
        try:
            # Check required fields
            required_fields = ['source', 'timestamp', 'data']
            if not all(field in data for field in required_fields):
                return False

            # Check timestamp format
            try:
                datetime.fromisoformat(data['timestamp'])
            except ValueError:
                return False

            # Check data is not empty
            if not data['data']:
                return False

            return True

        except Exception:
            return False

    async def _notify_callbacks(self, data: Dict[str, Any]):
        """Notify all registered callbacks of new data."""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    # Run synchronous callback in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, callback, data)
            except Exception as e:
                logger.error(f"Error in callback: {str(e)}")

    async def _handle_error(self, error: Exception, source_name: str):
        """Handle errors from data sources."""
        for error_handler in self.error_handlers:
            try:
                if asyncio.iscoroutinefunction(error_handler):
                    await error_handler(error, source_name)
                else:
                    # Run synchronous error handler in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, error_handler, error, source_name)
            except Exception as e:
                logger.error(f"Error in error handler: {str(e)}")

    async def _cleanup_loop(self):
        """Cleanup old data periodically."""
        while self.is_running:
            try:
                current_time = datetime.now()
                retention_hours = self.config['data_retention_hours']

                for source_name in self.data_history:
                    # Remove old data
                    cutoff_time = current_time - \
                        timedelta(hours=retention_hours)
                    self.data_history[source_name] = [
                        item for item in self.data_history[source_name]
                        if datetime.fromisoformat(item['timestamp']) > cutoff_time
                    ]

                # Wait before next cleanup
                await asyncio.sleep(3600)  # Cleanup every hour

            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(3600)

    def get_latest_data(self, source_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get latest data from specified source or all sources.

        Args:
            source_name: Name of specific source, or None for all sources

        Returns:
            Dictionary containing latest data
        """
        if source_name:
            return self.latest_data.get(source_name, {})
        else:
            return self.latest_data.copy()

    def get_data_history(self, source_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get historical data from a specific source.

        Args:
            source_name: Name of the data source
            hours: Number of hours of history to retrieve

        Returns:
            List of historical data items
        """
        if source_name not in self.data_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            item for item in self.data_history[source_name]
            if datetime.fromisoformat(item['timestamp']) > cutoff_time
        ]

    def get_connection_status(self) -> Dict[str, str]:
        """Get connection status for all sources."""
        return self.connection_status.copy()

    def get_queue_status(self) -> Dict[str, int]:
        """Get queue sizes for all sources."""
        return {
            source_name: queue.qsize()
            for source_name, queue in self.data_queues.items()
        }


class FinancialDataStreamer(RealTimeDataStreamer):
    """
    Specialized financial data streamer with built-in financial data processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the financial data streamer."""
        super().__init__(config)

        # Add financial data processors
        self.add_data_processor('financial_validation',
                                self._validate_financial_data)
        self.add_data_processor('price_normalization', self._normalize_prices)
        self.add_data_processor('technical_indicators',
                                self._calculate_technical_indicators)

    async def _validate_financial_data(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Validate financial data quality."""
        try:
            if 'data' in data_item and isinstance(data_item['data'], dict):
                financial_data = data_item['data']

                # Check for required financial fields
                required_fields = ['price', 'volume', 'timestamp']
                if all(field in financial_data for field in required_fields):
                    # Validate price is positive
                    if financial_data['price'] <= 0:
                        data_item['data_quality'] = 'invalid_price'
                        return data_item

                    # Validate volume is non-negative
                    if financial_data['volume'] < 0:
                        data_item['data_quality'] = 'invalid_volume'
                        return data_item

                    data_item['data_quality'] = 'valid'
                else:
                    data_item['data_quality'] = 'missing_fields'

            return data_item

        except Exception as e:
            logger.error(f"Error validating financial data: {str(e)}")
            data_item['data_quality'] = 'validation_error'
            return data_item

    async def _normalize_prices(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize price data to consistent format."""
        try:
            if 'data' in data_item and isinstance(data_item['data'], dict):
                financial_data = data_item['data']

                # Ensure price is float
                if 'price' in financial_data:
                    financial_data['price'] = float(financial_data['price'])

                # Ensure volume is integer
                if 'volume' in financial_data:
                    financial_data['volume'] = int(financial_data['volume'])

                # Add normalized timestamp
                if 'timestamp' in financial_data:
                    try:
                        timestamp = pd.to_datetime(financial_data['timestamp'])
                        financial_data['normalized_timestamp'] = timestamp.isoformat(
                        )
                    except:
                        financial_data['normalized_timestamp'] = data_item['timestamp']

            return data_item

        except Exception as e:
            logger.error(f"Error normalizing prices: {str(e)}")
            return data_item

    async def _calculate_technical_indicators(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic technical indicators for financial data."""
        try:
            if 'data' in data_item and isinstance(data_item['data'], dict):
                financial_data = data_item['data']

                # This is a simplified version - in practice, you'd want more sophisticated calculations
                if 'price' in financial_data and 'volume' in financial_data:
                    # Add basic indicators
                    financial_data['price_volume_ratio'] = financial_data['price'] / \
                        max(financial_data['volume'], 1)

                    # Add timestamp-based indicators
                    if 'normalized_timestamp' in financial_data:
                        timestamp = pd.to_datetime(
                            financial_data['normalized_timestamp'])
                        financial_data['hour_of_day'] = timestamp.hour
                        financial_data['day_of_week'] = timestamp.dayofweek

            return data_item

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return data_item
